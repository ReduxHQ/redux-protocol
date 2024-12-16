import { Tweet } from "agent-twitter-client";
import {
    composeContext,
    generateText,
    getEmbeddingZeroVector,
    IAgentRuntime,
    ModelClass,
    stringToUuid,
    parseBooleanFromText,
    ModelProviderName,
} from "@ai16z/eliza";
import { elizaLogger } from "@ai16z/eliza";
import { ClientBase } from "./base.ts";
import {
    agentSettingQueries,
    logQueries,
    tweetQueries,
} from "../../redux-extensions/src/db/queries";
import { Tweet as DBTweet } from "../../redux-extensions/src/db/schema";
import { promptQueries } from "../../redux-extensions/src/db/queries";
import { getNextTweetType } from "../../redux-extensions/src/twitter-extensions/weight-manager.ts";
import { generateMediaTweet } from "../../redux-extensions/src/engine/media-generation.ts";
import { urlToImage } from "../../redux-extensions/src/storage/cdn.ts";
import {
    twitterPostTemplate,
    twitterPostTemplateImg,
    twitterPostTemplateV1,
} from "../../redux-extensions/src/twitter-extensions/prompts.ts";

const MAX_TWEET_LENGTH = 280;

/**
 * Truncate text to fit within the Twitter character limit, ensuring it ends at a complete sentence.
 */
function truncateToCompleteSentence(text: string): string {
    if (text.length <= MAX_TWEET_LENGTH) {
        return text;
    }

    // Attempt to truncate at the last period within the limit
    const truncatedAtPeriod = text.slice(
        0,
        text.lastIndexOf(".", MAX_TWEET_LENGTH) + 1
    );
    if (truncatedAtPeriod.trim().length > 0) {
        return truncatedAtPeriod.trim();
    }

    // If no period is found, truncate to the nearest whitespace
    const truncatedAtSpace = text.slice(
        0,
        text.lastIndexOf(" ", MAX_TWEET_LENGTH)
    );
    if (truncatedAtSpace.trim().length > 0) {
        return truncatedAtSpace.trim() + "...";
    }

    // Fallback: Hard truncate and add ellipsis
    return text.slice(0, MAX_TWEET_LENGTH - 3).trim() + "...";
}

export class TwitterPostClient {
    client: ClientBase;
    runtime: IAgentRuntime;

    constructor(client: ClientBase, runtime: IAgentRuntime) {
        this.client = client;
        this.runtime = runtime;
    }

    /**
     * Generate a random future date for a tweet
     * @returns a Date object
     */
    async randomFutureDate() {
        const lastPost = await tweetQueries.getPendingTweets();
        // create a future date that is 10min - 50min from last post
        const lastPostTimestamp = lastPost[0]?.createdAt ?? new Date();
        const minMinutes =
            parseInt(this.runtime.getSetting("POST_INTERVAL_MIN")) || 12;
        const maxMinutes =
            parseInt(this.runtime.getSetting("POST_INTERVAL_MAX")) || 24;
        const randomMinutes =
            Math.floor(Math.random() * (maxMinutes - minMinutes + 120)) +
            minMinutes;
        const futureDate = new Date(
            lastPostTimestamp.getTime() + randomMinutes * 60 * 1000
        );
        return futureDate;
    }

    private async checkAndSendApprovedTweets() {
        elizaLogger.log(
            "Checking and sending approved tweets for posting (runs every 5 minutes)"
        );
        try {
            try {
                const result = await tweetQueries.getApprovedTweets();
                if (result.length === 0) {
                    elizaLogger.log("No approved tweets to send");
                    return;
                }
                for (const tweet of result) {
                    if (this.runtime.getSetting("TWITTER_DRY_RUN") === "true") {
                        elizaLogger.info(
                            `Dry run: would have posted tweet: ${tweet.content}`
                        );
                        continue;
                    }

                    const mediaData = tweet.mediaUrl
                        ? {
                              data: await urlToImage(tweet.mediaUrl),
                              mediaType: tweet.mediaType,
                          }
                        : null;

                    const homeTimeline: Partial<Tweet>[] =
                        tweet.homeTimeline as Partial<Tweet>[];

                    // before sending check that we are not sending a duplicate tweet
                    const tweetInDB = await tweetQueries.getSentTweetById(
                        tweet.id
                    );
                    const sent = tweetInDB
                        .map((t) => t.status)
                        .includes("sent");
                    if (sent) {
                        elizaLogger.log(
                            `Duplicate tweet found: ${tweet.id} ${tweet.status}`
                        );
                        continue;
                    }
                    // mark as sent now
                    await tweetQueries.markTweetAsSent(tweet.id);
                    try {
                        await this.sendTweetAndUpdateCache(
                            tweet.content,
                            homeTimeline,
                            tweet.newTweetContent,
                            tweet.id,
                            mediaData
                        );
                    } catch (error) {
                        elizaLogger.error(
                            "Failed to send tweet in update cache:",
                            error
                        );
                        await tweetQueries.markTweetAsError(
                            tweet.id,
                            JSON.stringify(error)
                        );
                    }
                    // errors will be logged in sendTweetAndUpdateCache
                    // set timeout here to avoid rate limits
                    await new Promise((resolve) => setTimeout(resolve, 60000));
                }
            } catch (error) {
                elizaLogger.error("Error checking approved tweets: ", error);
                // log error to admin log
                await logQueries.saveLog({
                    id: stringToUuid(
                        `twitter_send_approved_error_${new Date().getTime()}`
                    ),
                    userId: this.runtime.agentId,
                    body: { error },
                    type: "twitter",
                    createdAt: new Date(),
                    roomId: stringToUuid("twitter_send_approved_error_room"),
                });
                // mark as error
            } finally {
                elizaLogger.log("Finished checking approved tweets");
            }
        } catch (error) {
            elizaLogger.error(`Error checking approved tweets: ${error}`);
        }
    }

    private async saveTweetToDb({
        content,
        context,
        homeTimeline,
        newTweetContent,
        scheduledFor,
        mediaType,
        mediaUrl,
    }: {
        content: string;
        context: string;
        homeTimeline: Tweet[];
        newTweetContent: string;
        scheduledFor: Date | null;
        mediaType: string;
        mediaUrl: string | null;
    }) {
        scheduledFor = scheduledFor ?? (await this.randomFutureDate());
        elizaLogger.log(
            `Saving tweet to db: ${content} ${scheduledFor} ${mediaType} ${mediaUrl}`
        );
        // save tweet, context, homeTimeline, newTweetContent
        // homeTimeline and newTweetContent are required for the tweet to be posted later.
        try {
            const tweetToSave: DBTweet = {
                id: stringToUuid(`twitter_${new Date().getTime()}`),
                content,
                scheduledFor: scheduledFor,
                agentId: this.runtime.agentId,
                status: "pending",
                prompt: context,
                homeTimeline: JSON.stringify(homeTimeline),
                newTweetContent: newTweetContent,
                mediaType: mediaType,
                mediaUrl: mediaUrl,
                createdAt: new Date(),
                sentAt: null,
                error: null,
            };

            const result = await tweetQueries.saveTweetObject(tweetToSave);
            return result.id;
        } catch (error) {
            elizaLogger.error("Error saving tweet to database:", error);
            throw error;
        }
    }

    async getPostInterval(): Promise<number> {
        try {
            const result = await agentSettingQueries.getAgentSetting(
                this.runtime.agentId,
                "tweet_interval"
            );
            if (!result) {
                // create a new setting
                await agentSettingQueries.updateAgentSetting(
                    this.runtime.agentId,
                    "tweet_interval",
                    "30"
                );
            }
            return parseInt(result ?? "30") * 60 * 1000;
        } catch (error) {
            elizaLogger.error("Error getting post interval:", error);
            return 15 * 60 * 1000;
        }
    }

    async start(postImmediately: boolean = false) {
        if (!this.client.profile) {
            await this.client.init();
        }

        // new loop for approved tweets
        const startApprovedTweetsLoop = async () => {
            setTimeout(startApprovedTweetsLoop, 5 * 60 * 1000);
            await this.checkAndSendApprovedTweets();
        };

        startApprovedTweetsLoop();

        const generateNewTweetLoop = async () => {
            const interval = await this.getPostInterval();
            elizaLogger.info(
                `Next tweet will be generated in ${interval / (60 * 1000)} minutes`
            );

            setTimeout(async () => {
                await this.generateNewTweet();
                generateNewTweetLoop();
            }, interval);
        };

        if (
            this.runtime.getSetting("POST_IMMEDIATELY") != null &&
            this.runtime.getSetting("POST_IMMEDIATELY") != ""
        ) {
            postImmediately = parseBooleanFromText(
                this.runtime.getSetting("POST_IMMEDIATELY")
            );
        }
        if (postImmediately) {
            this.generateNewTweet();
        }

        generateNewTweetLoop();
    }

    private async sendTweetAndUpdateCache(
        content: string,
        homeTimeline: Partial<Tweet>[],
        newTweetContent: string,
        savedTweetId: string | null = null,
        mediaData: {
            data: Buffer;
            mediaType: string;
        } | null = null
    ): Promise<Tweet | null> {
        elizaLogger.error(`Posting new tweet`);
        let result: Response | null = null;
        try {
            result = await this.client.requestQueue.add(
                async () =>
                    await this.client.twitterClient.sendTweet(
                        content,
                        null,
                        mediaData ? [mediaData] : []
                    )
            );
            elizaLogger.error(`Tweet sent`);
        } catch (error) {
            elizaLogger.error(
                "Error sending tweet could not be queued or sent:",
                error
            );
            return null;
        }

        const body = await result.json();
        if (!body?.data?.create_tweet?.tweet_results?.result) {
            elizaLogger.error("Error sending tweet; Bad response:", body);
            // save error to db
            try {
                if (savedTweetId) {
                    await tweetQueries.updateTweetStatus(
                        savedTweetId,
                        "error",
                        JSON.stringify({
                            error: body?.data?.create_tweet?.tweet_results
                                ?.result?.errors,
                            tweet: body,
                        })
                    );
                }
                const logId = savedTweetId
                    ? stringToUuid(
                          `twitter_send_and_update_cache_error_${savedTweetId}_${new Date().getTime()}`
                      )
                    : stringToUuid(
                          `twitter_send_and_update_cache_error_${new Date().getTime()}`
                      );
                // log error to admin log
                await logQueries.saveLog({
                    id: logId,
                    userId: this.runtime.agentId,
                    body: body,
                    type: "twitter",
                    roomId: stringToUuid(
                        "twitter_send_and_update_cache_error_room"
                    ),
                    createdAt: new Date(),
                });
            } catch (error) {
                elizaLogger.error(
                    "Error logging error to admin log and tweet db:",
                    error
                );
            }
            return null;
        }
        const tweetResult = body.data.create_tweet.tweet_results.result;
        if (savedTweetId) {
            await tweetQueries.updateTweetStatus(savedTweetId, "sent");
        }

        const tweet = {
            id: tweetResult.rest_id,
            name: this.client.profile.screenName,
            username: this.client.profile.username,
            text: tweetResult.legacy.full_text,
            conversationId: tweetResult.legacy.conversation_id_str,
            createdAt: tweetResult.legacy.created_at,
            userId: this.client.profile.id,
            inReplyToStatusId: tweetResult.legacy.in_reply_to_status_id_str,
            permanentUrl: `https://twitter.com/${this.runtime.getSetting("TWITTER_USERNAME")}/status/${tweetResult.rest_id}`,
            hashtags: [],
            mentions: [],
            photos: tweetResult.photos,
            thread: [],
            urls: [],
            videos: tweetResult.videos,
        } as Tweet;

        await this.runtime.cacheManager.set(
            `twitter/${this.client.profile.username}/lastPost`,
            {
                id: tweet.id,
                timestamp: Date.now(),
            }
        );

        await this.client.cacheTweet(tweet);

        homeTimeline.push(tweet);
        await this.client.cacheTimeline(homeTimeline);
        elizaLogger.log(`Tweet posted:\n ${tweet.permanentUrl}`);

        const roomId = stringToUuid(
            tweet.conversationId + "-" + this.runtime.agentId
        );

        await this.runtime.ensureRoomExists(roomId);
        await this.runtime.ensureParticipantInRoom(
            this.runtime.agentId,
            roomId
        );

        await this.runtime.messageManager.createMemory({
            id: stringToUuid(tweet.id + "-" + this.runtime.agentId),
            userId: this.runtime.agentId,
            agentId: this.runtime.agentId,
            content: {
                text: newTweetContent.trim(),
                url: tweet.permanentUrl,
                source: "twitter",
            },
            roomId,
            embedding: getEmbeddingZeroVector(),
            createdAt: tweet.timestamp * 1000,
        });

        return tweet;
    }

    async getPrompt(tweetType: string) {
        // template type
        const promptKey =
            tweetType === "image/jpeg"
                ? "twitter_post_template_img"
                : "twitter_post_template";
        let prompt =
            tweetType === "image/jpeg"
                ? twitterPostTemplateImg
                : twitterPostTemplateV1;
        try {
            const dbTemplate = await promptQueries.getPrompt(
                this.runtime.agentId,
                promptKey
            );
            if (dbTemplate) {
                prompt = dbTemplate.prompt;
            }
        } catch (error) {
            elizaLogger.error("Error getting db template:", error);
        }
        return prompt;
    }

    private async generateNewTweet() {
        elizaLogger.log("Generating new tweet");

        try {
            await this.runtime.ensureUserExists(
                this.runtime.agentId,
                this.client.profile.username,
                this.runtime.character.name,
                "twitter"
            );

            let homeTimeline: Tweet[] = [];

            const cachedTimeline = await this.client.getCachedTimeline();

            if (cachedTimeline) {
                homeTimeline = cachedTimeline;
            } else {
                homeTimeline = await this.client.fetchHomeTimeline(20);
                await this.client.cacheTimeline(homeTimeline);
            }
            const formattedHomeTimeline =
                `# ${this.runtime.character.name}'s Home Timeline\n\n` +
                homeTimeline
                    .map((tweet) => {
                        return `#${tweet.id}\n${tweet.name} (@${tweet.username})${tweet.inReplyToStatusId ? `\nIn reply to: ${tweet.inReplyToStatusId}` : ""}\n${new Date(tweet.timestamp).toDateString()}\n\n${tweet.text}\n---\n`;
                    })
                    .join("\n");

            const topics = this.runtime.character.topics.join(", ");
            elizaLogger.log(`Topics: ${topics} now composing state`);
            const state = await this.runtime.composeState(
                {
                    userId: this.runtime.agentId,
                    roomId: stringToUuid("twitter_generate_room"),
                    agentId: this.runtime.agentId,
                    content: {
                        text: topics,
                        action: "",
                    },
                },
                {
                    twitterUserName: this.client.profile.username,
                    timeline: formattedHomeTimeline,
                }
            );
            elizaLogger.log("State composed. Now composing context.");

            const recentSentTweets = await tweetQueries.getSentTweets(
                this.runtime.agentId,
                15
            );
            const recentSentTweetsText = recentSentTweets
                .map((tweet) => {
                    return `#${tweet.id}\n${tweet.content}\n---\n`;
                })
                .join("\n");

            state.recentTweets = `\nHere are your recent tweets:\n${recentSentTweetsText}`;

            const tweetType = await getNextTweetType();
            const prompt = await this.getPrompt(tweetType);

            const context = composeContext({
                state,
                template: prompt,
            });
            elizaLogger.log("Context composed. Now generating tweet.");
            elizaLogger.debug("generate post prompt:\n" + context);

            let newTweetContent = "";
            let mediaUrl = null;
            if (tweetType === "image/jpeg") {
                const mediaTweet = await generateMediaTweet(context);
                newTweetContent = mediaTweet.tweetText;
                mediaUrl = mediaTweet.url;
            } else {
                newTweetContent = await generateText({
                    runtime: this.runtime,
                    context: context,
                    modelClass: ModelClass.LARGE,
                });
            }

            // Replace \n with proper line breaks and trim excess spaces
            const formattedTweet = newTweetContent
                .replaceAll(/\\n/g, "\n")
                .trim();

            // Use the helper function to truncate to complete sentence
            const content = truncateToCompleteSentence(formattedTweet);

            if (this.runtime.getSetting("TWITTER_DRY_RUN") === "true") {
                elizaLogger.info(
                    `Dry run: would have posted tweet: ${content}`
                );
                return;
            }

            // Check if approval is required
            elizaLogger.log("Approval is ALWAYS required. saving tweet to db");
            // if (this.runtime.getSetting("TWITTER_REQUIRE_APPROVAL") === "true") {
            // }
            try {
                const tweetId = await this.saveTweetToDb({
                    content,
                    context,
                    homeTimeline,
                    newTweetContent,
                    scheduledFor: null,
                    mediaType: tweetType,
                    mediaUrl,
                });
                elizaLogger.log(`Tweet saved for approval with ID: ${tweetId}`);
                return;
            } catch (error) {
                elizaLogger.error("Error saving tweet to database:", error);
                return;
            }
        } catch (error) {
            elizaLogger.error("Error generating new tweet:", error);
        }
    }
}
