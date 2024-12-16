import { AgentRuntime, elizaLogger } from "@ai16z/eliza";
import Instructor from "@instructor-ai/instructor";
import OpenAI from "openai";
import { z } from "zod";
import { tweetQueries } from "../db/queries";
import { DEFAULT_ADMIN_PROMPT, REVIEW_PROMPT } from "./prompts";

const TweetReviewSchema = z.object({
    status: z
        .enum(["admin_approved", "admin_rejected"])
        .describe("Status of the tweet"),
    isAppropriate: z
        .boolean()
        .describe("Whether the tweet is appropriate for posting"),
    confidence: z.number().describe("Confidence score between 0 and 1"),
    repetitionScore: z
        .number()
        .optional()
        .nullable()
        .describe(
            "Score between 0 and 100 indicating how similar the tweet is to previous tweets"
        ),
    reasoning: z
        .string()
        .describe("Detailed explanation of the review decision"),
    suggestedEdits: z
        .string()
        .optional()
        .nullable()
        .describe("Suggested improvements if needed"),
    evalLog: z
        .string()
        .optional()
        .nullable()
        .describe(
            "Evaluation log, include a short summary evaluation to improve prompts in the future."
        ),
    newTweet: z
        .string()
        .optional()
        .nullable()
        .describe("New tweet if suggested edits are provided"),
});

export type TweetReview = z.infer<typeof TweetReviewSchema>;

export async function reviewTweet(
    tweet: string,
    runtime: AgentRuntime,
    recentTweetsText: string
) {
    const oai = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY,
    });
    const reviewPrompt = REVIEW_PROMPT.replace("{tweet}", tweet).replace(
        "{recentTweets}",
        recentTweetsText
    );
    const client = Instructor({
        client: oai,
        mode: "FUNCTIONS",
    });

    const prompt = `${DEFAULT_ADMIN_PROMPT}

    ${reviewPrompt}
`;

    try {
        const review = await client.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            model: "gpt-4o-mini",
            response_model: {
                schema: TweetReviewSchema,
                name: "TweetReview",
            },
        });
        elizaLogger.info(`Review: ${JSON.stringify(review)}`);
        return review;
    } catch (error) {
        elizaLogger.error("Error reviewing tweet:", error);
        throw error;
    }
}

export async function updateTweetStatus(tweetId: string, review) {
    // once approved tweets are automatically posted.
    const status = review.status === "admin_approved" ? "approved" : "rejected";
    const reviewNotes = `
        Status: ${review.status}
        Confidence: ${review.confidence}
        Reasoning: ${review.reasoning}
        ${review.suggestedEdits ? `Suggested Edits: ${review.suggestedEdits}` : ""}
        ${review.newTweet ? `New Tweet: ${review.newTweet}` : ""}
    `;
    elizaLogger.info(`Review notes for tweet ${tweetId}: ${reviewNotes}`);
    await tweetQueries.updateTweetStatus(tweetId, status, reviewNotes);
}

export async function startTweetReviewEngine(runtime: AgentRuntime) {
    elizaLogger.info("Starting tweet review engine...");

    const reviewTweets = async () => {
        const pendingTweets = await tweetQueries.getPendingTweets();
        if (pendingTweets.length === 0) {
            elizaLogger.info(
                "No pending tweets found. Waiting for new tweets..."
            );
            return;
        }
        elizaLogger.info(
            `Found ${pendingTweets.length} pending tweets for review. Reviewing...`
        );

        const recentTweets = await tweetQueries.getSentTweets(
            pendingTweets[0].agentId,
            20
        );
        const recentTweetsText = recentTweets
            .map((tweet) => {
                return `#${tweet.id}\n${tweet.content}\n---\n`;
            })
            .join("\n");
        for (const tweet of pendingTweets) {
            try {
                const review = await reviewTweet(
                    tweet.content,
                    runtime,
                    recentTweetsText
                );
                await updateTweetStatus(tweet.id, review);

                // If there's a new suggested tweet, save it
                if (review.newTweet && review.status === "admin_rejected") {
                    await tweetQueries.saveTweet(
                        review.newTweet,
                        tweet.agentId,
                        tweet.scheduledFor
                    );
                }

                elizaLogger.info(
                    `Tweet ${tweet.id} reviewed: ${review.status}`
                );
            } catch (error) {
                elizaLogger.error(`Error reviewing tweet ${tweet.id}:`, error);
            }
        }
    };

    // Initial review
    await reviewTweets();

    // Set up interval (5 minutes)
    const interval = 5 * 60 * 1000;
    setInterval(reviewTweets, interval);
}
