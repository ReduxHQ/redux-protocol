import { db } from "./index";
import {
    tweets,
    agentSettings,
    agentPrompts,
    streamSettings,
    consciousnessStreams,
    Log,
    logs,
    cache,
} from "./schema";
import {
    Tweet,
    AgentPrompt,
    AgentSetting,
    StreamSetting,
    ConsciousnessStream,
    Cache,
} from "./schema";
import { eq, and, or, lte, desc, inArray, gte } from "drizzle-orm";
import { elizaLogger, stringToUuid } from "@ai16z/eliza";
import { v4 as uuidv4 } from "uuid";

export const tweetQueries = {
    getPendingTweets: async (): Promise<Tweet[]> => {
        try {
            return await db
                .select()
                .from(tweets)
                .where(and(eq(tweets.status, "pending")))
                .orderBy(desc(tweets.createdAt));
        } catch (error) {
            elizaLogger.error("Error fetching pending tweets:", error);
            throw error;
        }
    },

    getSentTweets: async (agentId: string, limit: number): Promise<Tweet[]> => {
        return await db
            .select()
            .from(tweets)
            .where(and(eq(tweets.agentId, agentId), eq(tweets.status, "sent")))
            .orderBy(desc(tweets.createdAt))
            .limit(limit);
    },

    getTweets: async (limit: number, offset: number): Promise<Tweet[]> => {
        return await db
            .select()
            .from(tweets)
            .orderBy(desc(tweets.createdAt))
            .limit(limit)
            .offset(offset);
    },
    updateTweetStatus: async (
        tweetId: string,
        status: string,
        error?: string
    ) => {
        try {
            await db
                .update(tweets)
                .set({ status, error } as Partial<Tweet>)
                .where(eq(tweets.id, tweetId));
        } catch (error) {
            elizaLogger.error("Error updating tweet status:", error);
            throw error;
        }
    },
    saveTweetObject: async (tweet: Tweet) => {
        try {
            const result = await db
                .insert(tweets)
                .values(tweet)
                .returning({ id: tweets.id });
            return result[0];
        } catch (error) {
            elizaLogger.error("Error saving tweet:", error);
            throw error;
        }
    },
    saveTweet: async (
        content: string,
        agentId: string,
        scheduledFor?: Date,
        homeTimeline?: Tweet[],
        newTweetContent?: string
    ) => {
        const tweet: Tweet = {
            id: uuidv4(),
            content,
            agentId,
            scheduledFor,
            status: "pending",
            homeTimeline,
            newTweetContent,
            createdAt: new Date(),
            sentAt: null,
            error: null,
            prompt: null,
            mediaType: "text",
            mediaUrl: null,
        };
        try {
            const result = await db.insert(tweets).values(tweet).returning();
            return result[0];
        } catch (error) {
            elizaLogger.error("Error saving tweet:", error);
            throw error;
        }
    },

    getApprovedTweets: async () => {
        try {
            return await db
                .select()
                .from(tweets)
                .where(
                    and(
                        eq(tweets.status, "approved"),
                        or(
                            eq(tweets.scheduledFor, null),
                            lte(tweets.scheduledFor, new Date())
                        )
                    )
                )
                .orderBy(desc(tweets.createdAt));
        } catch (error) {
            elizaLogger.error("Error fetching approved tweets:", error);
            throw error;
        }
    },

    markTweetAsSent: async (tweetId: string) => {
        try {
            await db
                .update(tweets)
                .set({ status: "sent", sentAt: new Date() } as Partial<Tweet>)
                .where(eq(tweets.id, tweetId));
        } catch (error) {
            elizaLogger.error("Error marking tweet as sent:", error);
            throw error;
        }
    },

    markTweetAsError: async (tweetId: string, error: string) => {
        try {
            await db
                .update(tweets)
                .set({
                    status: "error",
                    error,
                } as Partial<Tweet>)
                .where(eq(tweets.id, tweetId));
        } catch (error) {
            elizaLogger.error("Error marking tweet as error:", error);
            throw error;
        }
    },

    getSentTweetById: async (tweetId: string) => {
        try {
            return await db
                .select()
                .from(tweets)
                .where(and(eq(tweets.id, tweetId), eq(tweets.status, "sent")));
        } catch (error) {
            elizaLogger.error("Error fetching sent tweet:", error);
            throw error;
        }
    },
    // bulk update as sending
    updateTweetsAsSending: async (tweetIds: string[]) => {
        await db
            .update(tweets)
            .set({ status: "sending" } as Partial<Tweet>)
            .where(inArray(tweets.id, tweetIds));
    },
};

export const agentSettingQueries = {
    getAgentSetting: async (
        agentId: string,
        key: string
    ): Promise<string | undefined> => {
        try {
            const result: AgentSetting[] = await db
                .select()
                .from(agentSettings)
                .where(
                    and(
                        eq(agentSettings.agentId, agentId),
                        eq(agentSettings.settingKey, key)
                    )
                )
                .limit(1);

            return result[0]?.settingValue;
        } catch (error) {
            elizaLogger.error("Error fetching agent setting:", error);
            throw error;
        }
    },

    updateAgentSetting: async (id: string, key: string, value: string) => {
        try {
            const setting: AgentSetting = {
                id: uuidv4(),
                agentId: id,
                settingKey: key,
                settingValue: value,
                createdAt: new Date(),
                updatedAt: new Date(),
            };
            await db
                .insert(agentSettings)
                .values(setting)
                .onConflictDoUpdate({
                    target: agentSettings.id,
                    set: {
                        settingValue: value,
                        updatedAt: new Date(),
                    } as Partial<AgentSetting>,
                });
        } catch (error) {
            elizaLogger.error("Error updating agent setting:", error);
            throw error;
        }
    },
};

export const promptQueries = {
    savePrompt: async (settings: AgentPrompt): Promise<AgentPrompt> => {
        try {
            const prompt: AgentPrompt = {
                id: uuidv4(),
                prompt: settings.prompt,
                agentId: settings.agentId,
                version: settings.version,
                enabled: settings.enabled,
                type: settings.type,
            };
            const result: AgentPrompt[] = await db
                .insert(agentPrompts)
                .values(prompt)
                .returning();
            return result[0];
        } catch (error) {
            elizaLogger.error("Error saving prompt:", error);
            throw error;
        }
    },

    getPrompt: async (
        agentId: string,
        type: string
    ): Promise<AgentPrompt | null> => {
        try {
            const result: AgentPrompt[] | undefined = await db
                .select()
                .from(agentPrompts)
                .where(
                    and(
                        eq(agentPrompts.agentId, agentId),
                        eq(agentPrompts.type, type),
                        eq(agentPrompts.enabled, true)
                    )
                )
                .limit(1);

            return result[0] || null;
        } catch (error) {
            elizaLogger.error("Error getting prompt:", error);
            throw error;
        }
    },

    updatePrompt: async (
        id: string,
        settings: Partial<AgentPrompt>
    ): Promise<AgentPrompt> => {
        try {
            const result: AgentPrompt[] = await db
                .update(agentPrompts)
                .set({
                    ...settings,
                    updatedAt: new Date(),
                } as Partial<AgentPrompt>)
                .where(eq(agentPrompts.id, id))
                .returning();
            return result[0];
        } catch (error) {
            elizaLogger.error("Error updating prompt:", error);
            throw error;
        }
    },

    getPrompts: async (): Promise<AgentPrompt[]> => {
        return await db.select().from(agentPrompts);
    },

    createPrompt: async (settings: AgentPrompt): Promise<AgentPrompt> => {
        try {
            const result: AgentPrompt[] = await db
                .insert(agentPrompts)
                .values(settings)
                .returning();
            return result[0];
        } catch (error) {
            elizaLogger.error("Error creating prompt:", error);
            throw error;
        }
    },

    deletePrompt: async (id: string): Promise<void> => {
        try {
            await db.delete(agentPrompts).where(eq(agentPrompts.id, id));
        } catch (error) {
            elizaLogger.error("Error deleting prompt:", error);
            throw error;
        }
    },
};

export const streamQueries = {
    getStreamSettings: async (agentId: string): Promise<StreamSetting> => {
        try {
            const result: StreamSetting[] = await db
                .select()
                .from(streamSettings)
                .where(eq(streamSettings.agentId, agentId));

            if (result.length === 0) {
                const defaultSettings: StreamSetting = {
                    id: uuidv4(),
                    agentId,
                    enabled: true,
                    interval: 15,
                    lastRun: new Date(),
                };
                await streamQueries.saveStreamSettings(defaultSettings);
                return defaultSettings;
            }

            return result[0];
        } catch (error) {
            elizaLogger.error("Error fetching stream settings:", error);
            throw error;
        }
    },

    saveStreamSettings: async (settings: StreamSetting): Promise<void> => {
        try {
            await db
                .insert(streamSettings)
                .values({
                    id: settings.id,
                    agentId: settings.agentId,
                    enabled: settings.enabled,
                    interval: settings.interval,
                    lastRun: settings.lastRun,
                })
                .onConflictDoUpdate({
                    target: streamSettings.id,
                    set: {
                        enabled: settings.enabled,
                        interval: settings.interval,
                        lastRun: settings.lastRun,
                    },
                });
        } catch (error) {
            elizaLogger.error("Error saving stream settings:", error);
            throw error;
        }
    },

    getRecentStreams: async (agentId: string, limit: number = 100) => {
        try {
            return await db
                .select()
                .from(consciousnessStreams)
                .where(eq(consciousnessStreams.agentId, agentId))
                .orderBy(desc(consciousnessStreams.timestamp))
                .limit(limit);
        } catch (error) {
            elizaLogger.error("Error fetching recent streams:", error);
            throw error;
        }
    },

    saveStream: async (
        entry: ConsciousnessStream,
        agentId?: string
    ): Promise<void> => {
        const stream: ConsciousnessStream = {
            id: uuidv4(),
            agentId: agentId ?? "davinci",
            topic: entry.topic,
            title: entry.title,
            content: entry.content,
            status: entry.status,
            timestamp: new Date(),
        };
        try {
            await db.insert(consciousnessStreams).values(stream);
        } catch (error) {
            elizaLogger.error("Error saving stream entry:", error);
            throw error;
        }
    },
};

export const logQueries = {
    saveLog: async (newLog: Log) => {
        await db.insert(logs).values(newLog);
    },

    // create a log for a prompt
    createPromptLog: async (
        logBody: object,
        agentId: string,
        promptType: string
    ) => {
        const log: Log = {
            id: stringToUuid(
                `prompt_log_${promptType}_${new Date().getTime()}`
            ),
            userId: agentId,
            body: JSON.stringify(logBody),
            type: "prompt",
            createdAt: new Date(),
            roomId: `prompt_log_${promptType}`,
        };
        await db.insert(logs).values(log);
    },
};

export const cacheQueries = {
    get: async (key: string) => {
        const result: Cache[] = await db
            .select()
            .from(cache)
            .where(and(eq(cache.key, key), gte(cache.expiresAt, new Date())))
            .orderBy(desc(cache.createdAt))
            .limit(1);
        return result;
    },
    set: async (key: string, value: string) => {
        const c: Cache = {
            key,
            value,
            createdAt: new Date(),
            expiresAt: new Date(Date.now() + 1000 * 60 * 60), // 1
            agentId: "davinci",
        };
        // check if the cache already exists
        const existingCache = await cacheQueries.get(key);
        if (existingCache && existingCache.length > 0) {
            await cacheQueries.delete(key);
        }
        await db.insert(cache).values(c);
    },
    delete: async (key: string) => {
        await db.delete(cache).where(eq(cache.key, key));
    },
};
