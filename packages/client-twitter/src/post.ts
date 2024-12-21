import { Tweet } from "agent-twitter-client";
import {
    composeContext,
    generateText,
    getEmbeddingZeroVector,
    IAgentRuntime,
    ModelClass,
    stringToUuid,
    parseBooleanFromText,
} from "@ai16z/eliza";
import { elizaLogger } from "@ai16z/eliza";
import { ClientBase } from "./base.ts";
import { postActionResponseFooter } from "@ai16z/eliza";
import { generateTweetActions } from "@ai16z/eliza";
import { IImageDescriptionService, ServiceType } from "@ai16z/eliza";
import { buildConversationThread } from "./utils.ts";
import { twitterMessageHandlerTemplate } from "./interactions.ts";
import {
    twitterPostTemplate,
    twitterPostTemplateImg,
    twitterPostTemplateV1,
} from "../../redux-extensions/src/twitter-extensions/prompts.ts";
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

const MAX_TWEET_LENGTH = 240;

export const twitterActionTemplate =
    `
# INSTRUCTIONS: Determine actions for {{agentName}} (@{{twitterUserName}}) based on:
{{bio}}
{{postDirections}}

Guidelines:
- Highly selective engagement
- Direct mentions are priority
- Skip: low-effort content, off-topic, repetitive

Actions (respond only with tags):
[LIKE] - Resonates with interests (9.5/10)
[RETWEET] - Perfect character alignment (9/10)
[QUOTE] - Can add unique value (8/10)
[REPLY] - Memetic opportunity (9/10)

Tweet:
{{currentTweet}}

# Respond with qualifying action tags only.` + postActionResponseFooter;

/**
 * Truncate text to fit within the Twitter character limit, ensuring it ends at a complete sentence.
 */
function truncateToCompleteSentence(
    text: string,
    maxTweetLength: number
): string {
    if (text.length <= maxTweetLength) {
        return text;
    }

    // Attempt to truncate at the last period within the limit
    const truncatedAtPeriod = text.slice(
        0,
        text.lastIndexOf(".", maxTweetLength) + 1
    );
    if (truncatedAtPeriod.trim().length > 0) {
        return truncatedAtPeriod.trim();
    }

    // If no period is found, truncate to the nearest whitespace
    const truncatedAtSpace = text.slice(
        0,
        text.lastIndexOf(" ", maxTweetLength)
    );
    if (truncatedAtSpace.trim().length > 0) {
        return truncatedAtSpace.trim() + "...";
    }

    // Fallback: Hard truncate and add ellipsis
    return text.slice(0, maxTweetLength - 3).trim() + "...";
}

export class TwitterPostClient {
    client: ClientBase;
    runtime: IAgentRuntime;
    twitterUsername: string;
    private isProcessing: boolean = false;
    private lastProcessTime: number = 0;
    private stopProcessingActions: boolean = false;

    constructor(client: ClientBase, runtime: IAgentRuntime) {
        this.client = client;
        this.runtime = runtime;
        this.twitterUsername = runtime.getSetting("TWITTER_USERNAME");
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

        const processActionsLoop = async () => {
            const actionInterval =
                parseInt(this.runtime.getSetting("ACTION_INTERVAL")) || 300000; // Default to 5 minutes

            while (!this.stopProcessingActions) {
                try {
                    const results = await this.processTweetActions();
                    if (results) {
                        elizaLogger.log(`Processed ${results.length} tweets`);
                        elizaLogger.log(
                            `Next action processing scheduled in ${actionInterval / 1000} seconds`
                        );
                        // Wait for the full interval before next processing
                        await new Promise((resolve) =>
                            setTimeout(resolve, actionInterval)
                        );
                    }
                } catch (error) {
                    elizaLogger.error(
                        "Error in action processing loop:",
                        error
                    );
                    // Add exponential backoff on error
                    await new Promise((resolve) => setTimeout(resolve, 30000)); // Wait 30s on error
                }
            }
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
            await this.generateNewTweet();
        }
        generateNewTweetLoop();

        // Add check for ENABLE_ACTION_PROCESSING before starting the loop
        const enableActionProcessing = parseBooleanFromText(
            this.runtime.getSetting("ENABLE_ACTION_PROCESSING") ?? "true"
        );

        if (enableActionProcessing) {
            processActionsLoop().catch((error) => {
                elizaLogger.error(
                    "Fatal error in process actions loop:",
                    error
                );
            });
        } else {
            elizaLogger.log("Action processing loop disabled by configuration");
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
            const roomId = stringToUuid(
                "twitter_generate_room-" + this.client.profile.username
            );
            await this.runtime.ensureUserExists(
                this.runtime.agentId,
                this.client.profile.username,
                this.runtime.character.name,
                "twitter"
            );

            // let homeTimeline: Tweet[] = [];

            // const cachedTimeline = await this.client.getCachedTimeline();

            // if (cachedTimeline) {
            //     homeTimeline = cachedTimeline;
            // } else {
            //     homeTimeline = await this.client.fetchHomeTimeline(20);
            //     await this.client.cacheTimeline(homeTimeline);
            // }
            // const formattedHomeTimeline =
            //     `# ${this.runtime.character.name}'s Home Timeline\n\n` +
            //     homeTimeline
            //         .map((tweet) => {
            //             return `#${tweet.id}\n${tweet.name} (@${tweet.username})${tweet.inReplyToStatusId ? `\nIn reply to: ${tweet.inReplyToStatusId}` : ""}\n${new Date(tweet.timestamp).toDateString()}\n\n${tweet.text}\n---\n`;
            //         })
            //         .join("\n");

            const topics = this.runtime.character.topics.join(", ");
            elizaLogger.log(`Topics: ${topics} now composing state`);

            const state = await this.runtime.composeState(
                {
                    userId: this.runtime.agentId,
                    roomId: stringToUuid("twitter_generate_room"),
                    agentId: this.runtime.agentId,
                    content: {
                        text: topics || "",
                        action: "TWEET",
                    },
                },
                {
                    twitterUserName: this.client.profile.username,
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

            // First attempt to clean content
            let cleanedContent = "";

            // Try parsing as JSON first
            try {
                const parsedResponse = JSON.parse(newTweetContent);
                if (parsedResponse.text) {
                    cleanedContent = parsedResponse.text;
                } else if (typeof parsedResponse === "string") {
                    cleanedContent = parsedResponse;
                }
            } catch (error) {
                error.linted = true; // make linter happy since catch needs a variable
                // If not JSON, clean the raw content
                cleanedContent = newTweetContent
                    .replace(/^\s*{?\s*"text":\s*"|"\s*}?\s*$/g, "") // Remove JSON-like wrapper
                    .replace(/^['"](.*)['"]$/g, "$1") // Remove quotes
                    .replace(/\\"/g, '"') // Unescape quotes
                    .replace(/\\n/g, "\n") // Unescape newlines
                    .trim();
            }

            if (!cleanedContent) {
                elizaLogger.error(
                    "Failed to extract valid content from response:",
                    {
                        rawResponse: newTweetContent,
                        attempted: "JSON parsing",
                    }
                );
                return;
            }

            // Use the helper function to truncate to complete sentence
            //const content = truncateToCompleteSentence(formattedTweet);
            const content = truncateToCompleteSentence(
                cleanedContent,
                MAX_TWEET_LENGTH
            );

            const removeQuotes = (str: string) =>
                str.replace(/^['"](.*)['"]$/, "$1");

            const fixNewLines = (str: string) => str.replaceAll(/\\n/g, "\n");

            // Final cleaning
            cleanedContent = removeQuotes(fixNewLines(content));

            if (this.runtime.getSetting("TWITTER_DRY_RUN") === "true") {
                elizaLogger.info(
                    `Dry run: would have posted tweet: ${cleanedContent}`
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
                    homeTimeline: [],
                    newTweetContent,
                    scheduledFor: null,
                    mediaType: tweetType,
                    mediaUrl,
                });

                elizaLogger.log(`Posting new tweet:\n ${cleanedContent}`);

                const result = await this.client.requestQueue.add(
                    async () =>
                        await this.client.twitterClient.sendTweet(
                            cleanedContent
                        )
                );
                const body = await result.json();
                if (!body?.data?.create_tweet?.tweet_results?.result) {
                    console.error("Error sending tweet; Bad response:", body);
                    return;
                }
                const tweetResult = body.data.create_tweet.tweet_results.result;

                const tweet = {
                    id: tweetResult.rest_id,
                    name: this.client.profile.screenName,
                    username: this.client.profile.username,
                    text: tweetResult.legacy.full_text,
                    conversationId: tweetResult.legacy.conversation_id_str,
                    createdAt: tweetResult.legacy.created_at,
                    timestamp: new Date(
                        tweetResult.legacy.created_at
                    ).getTime(),
                    userId: this.client.profile.id,
                    inReplyToStatusId:
                        tweetResult.legacy.in_reply_to_status_id_str,
                    permanentUrl: `https://twitter.com/${this.twitterUsername}/status/${tweetResult.rest_id}`,
                    hashtags: [],
                    mentions: [],
                    photos: [],
                    thread: [],
                    urls: [],
                    videos: [],
                } as Tweet;

                await this.runtime.cacheManager.set(
                    `twitter/${this.client.profile.username}/lastPost`,
                    {
                        id: tweet.id,
                        timestamp: Date.now(),
                    }
                );

                await this.client.cacheTweet(tweet);

                elizaLogger.log(`Tweet posted:\n ${tweet.permanentUrl}`);

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
                    createdAt: tweet.timestamp,
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

    private async generateTweetContent(
        tweetState: any,
        options?: {
            template?: string;
            context?: string;
        }
    ): Promise<string> {
        const context = composeContext({
            state: tweetState,
            template:
                options?.template ||
                this.runtime.character.templates?.twitterPostTemplate ||
                twitterPostTemplate,
        });

        const response = await generateText({
            runtime: this.runtime,
            context: options?.context || context,
            modelClass: ModelClass.SMALL,
        });
        console.log("generate tweet content response:\n" + response);

        // First clean up any markdown and newlines
        const cleanedResponse = response
            .replace(/```json\s*/g, "") // Remove ```json
            .replace(/```\s*/g, "") // Remove any remaining ```
            .replaceAll(/\\n/g, "\n")
            .trim();

        // Try to parse as JSON first
        try {
            const jsonResponse = JSON.parse(cleanedResponse);
            if (jsonResponse.text) {
                return this.trimTweetLength(jsonResponse.text);
            }
            if (typeof jsonResponse === "object") {
                const possibleContent =
                    jsonResponse.content ||
                    jsonResponse.message ||
                    jsonResponse.response;
                if (possibleContent) {
                    return this.trimTweetLength(possibleContent);
                }
            }
        } catch (error) {
            error.linted = true; // make linter happy since catch needs a variable

            // If JSON parsing fails, treat as plain text
            elizaLogger.debug("Response is not JSON, treating as plain text");
        }

        // If not JSON or no valid content found, clean the raw text
        return this.trimTweetLength(cleanedResponse);
    }

    // Helper method to ensure tweet length compliance
    private trimTweetLength(text: string, maxLength: number = 280): string {
        if (text.length <= maxLength) return text;

        // Try to cut at last sentence
        const lastSentence = text.slice(0, maxLength).lastIndexOf(".");
        if (lastSentence > 0) {
            return text.slice(0, lastSentence + 1).trim();
        }

        // Fallback to word boundary
        return (
            text.slice(0, text.lastIndexOf(" ", maxLength - 3)).trim() + "..."
        );
    }

    private async processTweetActions() {
        if (this.isProcessing) {
            elizaLogger.log("Already processing tweet actions, skipping");
            return null;
        }

        try {
            this.isProcessing = true;
            this.lastProcessTime = Date.now();

            elizaLogger.log("Processing tweet actions");

            await this.runtime.ensureUserExists(
                this.runtime.agentId,
                this.twitterUsername,
                this.runtime.character.name,
                "twitter"
            );

            const homeTimeline = await this.client.fetchTimelineForActions(15);
            const results = [];

            for (const tweet of homeTimeline) {
                try {
                    // Skip if we've already processed this tweet
                    const memory =
                        await this.runtime.messageManager.getMemoryById(
                            stringToUuid(tweet.id + "-" + this.runtime.agentId)
                        );
                    if (memory) {
                        elizaLogger.log(
                            `Already processed tweet ID: ${tweet.id}`
                        );
                        continue;
                    }

                    const roomId = stringToUuid(
                        tweet.conversationId + "-" + this.runtime.agentId
                    );

                    const tweetState = await this.runtime.composeState(
                        {
                            userId: this.runtime.agentId,
                            roomId,
                            agentId: this.runtime.agentId,
                            content: { text: "", action: "" },
                        },
                        {
                            twitterUserName: this.twitterUsername,
                            currentTweet: `ID: ${tweet.id}\nFrom: ${tweet.name} (@${tweet.username})\nText: ${tweet.text}`,
                        }
                    );

                    const actionContext = composeContext({
                        state: tweetState,
                        template:
                            this.runtime.character.templates
                                ?.twitterActionTemplate ||
                            twitterActionTemplate,
                    });

                    const actionResponse = await generateTweetActions({
                        runtime: this.runtime,
                        context: actionContext,
                        modelClass: ModelClass.SMALL,
                    });

                    if (!actionResponse) {
                        elizaLogger.log(
                            `No valid actions generated for tweet ${tweet.id}`
                        );
                        continue;
                    }

                    const executedActions: string[] = [];

                    // Execute actions
                    if (actionResponse.like) {
                        try {
                            await this.client.twitterClient.likeTweet(tweet.id);
                            executedActions.push("like");
                            elizaLogger.log(`Liked tweet ${tweet.id}`);
                        } catch (error) {
                            elizaLogger.error(
                                `Error liking tweet ${tweet.id}:`,
                                error
                            );
                        }
                    }

                    if (actionResponse.retweet) {
                        try {
                            await this.client.twitterClient.retweet(tweet.id);
                            executedActions.push("retweet");
                            elizaLogger.log(`Retweeted tweet ${tweet.id}`);
                        } catch (error) {
                            elizaLogger.error(
                                `Error retweeting tweet ${tweet.id}:`,
                                error
                            );
                        }
                    }

                    if (actionResponse.quote) {
                        try {
                            // Build conversation thread for context
                            const thread = await buildConversationThread(
                                tweet,
                                this.client
                            );
                            const formattedConversation = thread
                                .map(
                                    (t) =>
                                        `@${t.username} (${new Date(t.timestamp * 1000).toLocaleString()}): ${t.text}`
                                )
                                .join("\n\n");

                            // Generate image descriptions if present
                            const imageDescriptions = [];
                            if (tweet.photos?.length > 0) {
                                elizaLogger.log(
                                    "Processing images in tweet for context"
                                );
                                for (const photo of tweet.photos) {
                                    const description = await this.runtime
                                        .getService<IImageDescriptionService>(
                                            ServiceType.IMAGE_DESCRIPTION
                                        )
                                        .describeImage(photo.url);
                                    imageDescriptions.push(description);
                                }
                            }

                            // Handle quoted tweet if present
                            let quotedContent = "";
                            if (tweet.quotedStatusId) {
                                try {
                                    const quotedTweet =
                                        await this.client.twitterClient.getTweet(
                                            tweet.quotedStatusId
                                        );
                                    if (quotedTweet) {
                                        quotedContent = `\nQuoted Tweet from @${quotedTweet.username}:\n${quotedTweet.text}`;
                                    }
                                } catch (error) {
                                    elizaLogger.error(
                                        "Error fetching quoted tweet:",
                                        error
                                    );
                                }
                            }

                            // Compose rich state with all context
                            const enrichedState =
                                await this.runtime.composeState(
                                    {
                                        userId: this.runtime.agentId,
                                        roomId: stringToUuid(
                                            tweet.conversationId +
                                                "-" +
                                                this.runtime.agentId
                                        ),
                                        agentId: this.runtime.agentId,
                                        content: {
                                            text: tweet.text,
                                            action: "QUOTE",
                                        },
                                    },
                                    {
                                        twitterUserName: this.twitterUsername,
                                        currentPost: `From @${tweet.username}: ${tweet.text}`,
                                        formattedConversation,
                                        imageContext:
                                            imageDescriptions.length > 0
                                                ? `\nImages in Tweet:\n${imageDescriptions.map((desc, i) => `Image ${i + 1}: ${desc}`).join("\n")}`
                                                : "",
                                        quotedContent,
                                    }
                                );

                            const quoteContent =
                                await this.generateTweetContent(enrichedState, {
                                    template:
                                        this.runtime.character.templates
                                            ?.twitterMessageHandlerTemplate ||
                                        twitterMessageHandlerTemplate,
                                });

                            if (!quoteContent) {
                                elizaLogger.error(
                                    "Failed to generate valid quote tweet content"
                                );
                                return;
                            }

                            elizaLogger.log(
                                "Generated quote tweet content:",
                                quoteContent
                            );

                            // Send the tweet through request queue
                            const result = await this.client.requestQueue.add(
                                async () =>
                                    await this.client.twitterClient.sendQuoteTweet(
                                        quoteContent,
                                        tweet.id
                                    )
                            );

                            const body = await result.json();

                            if (
                                body?.data?.create_tweet?.tweet_results?.result
                            ) {
                                elizaLogger.log(
                                    "Successfully posted quote tweet"
                                );
                                executedActions.push("quote");

                                // Cache generation context for debugging
                                await this.runtime.cacheManager.set(
                                    `twitter/quote_generation_${tweet.id}.txt`,
                                    `Context:\n${enrichedState}\n\nGenerated Quote:\n${quoteContent}`
                                );
                            } else {
                                elizaLogger.error(
                                    "Quote tweet creation failed:",
                                    body
                                );
                            }
                        } catch (error) {
                            elizaLogger.error(
                                "Error in quote tweet generation:",
                                error
                            );
                        }
                    }

                    if (actionResponse.reply) {
                        try {
                            await this.handleTextOnlyReply(
                                tweet,
                                tweetState,
                                executedActions
                            );
                        } catch (error) {
                            elizaLogger.error(
                                `Error replying to tweet ${tweet.id}:`,
                                error
                            );
                        }
                    }

                    // Add these checks before creating memory
                    await this.runtime.ensureRoomExists(roomId);
                    await this.runtime.ensureUserExists(
                        stringToUuid(tweet.userId),
                        tweet.username,
                        tweet.name,
                        "twitter"
                    );
                    await this.runtime.ensureParticipantInRoom(
                        this.runtime.agentId,
                        roomId
                    );

                    // Then create the memory
                    await this.runtime.messageManager.createMemory({
                        id: stringToUuid(tweet.id + "-" + this.runtime.agentId),
                        userId: stringToUuid(tweet.userId),
                        content: {
                            text: tweet.text,
                            url: tweet.permanentUrl,
                            source: "twitter",
                            action: executedActions.join(","),
                        },
                        agentId: this.runtime.agentId,
                        roomId,
                        embedding: getEmbeddingZeroVector(),
                        createdAt: tweet.timestamp * 1000,
                    });

                    results.push({
                        tweetId: tweet.id,
                        parsedActions: actionResponse,
                        executedActions,
                    });
                } catch (error) {
                    elizaLogger.error(
                        `Error processing tweet ${tweet.id}:`,
                        error
                    );
                    continue;
                }
            }

            return results; // Return results array to indicate completion
        } catch (error) {
            elizaLogger.error("Error in processTweetActions:", error);
            throw error;
        } finally {
            this.isProcessing = false;
        }
    }

    private async handleTextOnlyReply(
        tweet: Tweet,
        tweetState: any,
        executedActions: string[]
    ) {
        try {
            // Build conversation thread for context
            const thread = await buildConversationThread(tweet, this.client);
            const formattedConversation = thread
                .map(
                    (t) =>
                        `@${t.username} (${new Date(t.timestamp * 1000).toLocaleString()}): ${t.text}`
                )
                .join("\n\n");

            // Generate image descriptions if present
            const imageDescriptions = [];
            if (tweet.photos?.length > 0) {
                elizaLogger.log("Processing images in tweet for context");
                for (const photo of tweet.photos) {
                    const description = await this.runtime
                        .getService<IImageDescriptionService>(
                            ServiceType.IMAGE_DESCRIPTION
                        )
                        .describeImage(photo.url);
                    imageDescriptions.push(description);
                }
            }

            // Handle quoted tweet if present
            let quotedContent = "";
            if (tweet.quotedStatusId) {
                try {
                    const quotedTweet =
                        await this.client.twitterClient.getTweet(
                            tweet.quotedStatusId
                        );
                    if (quotedTweet) {
                        quotedContent = `\nQuoted Tweet from @${quotedTweet.username}:\n${quotedTweet.text}`;
                    }
                } catch (error) {
                    elizaLogger.error("Error fetching quoted tweet:", error);
                }
            }

            // Compose rich state with all context
            const enrichedState = await this.runtime.composeState(
                {
                    userId: this.runtime.agentId,
                    roomId: stringToUuid(
                        tweet.conversationId + "-" + this.runtime.agentId
                    ),
                    agentId: this.runtime.agentId,
                    content: { text: tweet.text, action: "" },
                },
                {
                    twitterUserName: this.twitterUsername,
                    currentPost: `From @${tweet.username}: ${tweet.text}`,
                    formattedConversation,
                    imageContext:
                        imageDescriptions.length > 0
                            ? `\nImages in Tweet:\n${imageDescriptions.map((desc, i) => `Image ${i + 1}: ${desc}`).join("\n")}`
                            : "",
                    quotedContent,
                }
            );

            // Generate and clean the reply content
            const replyText = await this.generateTweetContent(enrichedState, {
                template:
                    this.runtime.character.templates
                        ?.twitterMessageHandlerTemplate ||
                    twitterMessageHandlerTemplate,
            });

            if (!replyText) {
                elizaLogger.error("Failed to generate valid reply content");
                return;
            }

            elizaLogger.debug("Final reply text to be sent:", replyText);

            // Send the tweet through request queue
            const result = await this.client.requestQueue.add(
                async () =>
                    await this.client.twitterClient.sendTweet(
                        replyText,
                        tweet.id
                    )
            );

            const body = await result.json();

            if (body?.data?.create_tweet?.tweet_results?.result) {
                elizaLogger.log("Successfully posted reply tweet");
                executedActions.push("reply");

                // Cache generation context for debugging
                await this.runtime.cacheManager.set(
                    `twitter/reply_generation_${tweet.id}.txt`,
                    `Context:\n${enrichedState}\n\nGenerated Reply:\n${replyText}`
                );
            } else {
                elizaLogger.error("Tweet reply creation failed:", body);
            }
        } catch (error) {
            elizaLogger.error("Error in handleTextOnlyReply:", error);
        }
    }

    async stop() {
        this.stopProcessingActions = true;
    }
}
