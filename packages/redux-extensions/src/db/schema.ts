import { InferSelectModel } from "drizzle-orm";
import {
    pgTable,
    uuid,
    text,
    timestamp,
    jsonb,
    boolean,
    integer,
} from "drizzle-orm/pg-core";

export const accounts = pgTable("accounts", {
    id: uuid("id").primaryKey(),
    createdAt: timestamp("createdAt", { withTimezone: true }).defaultNow(),
    name: text("name"),
    username: text("username"),
    email: text("email").notNull(),
    avatarUrl: text("avatar_url"),
    details: jsonb("details").default({}),
});

export const agentSettings = pgTable("agent_settings", {
    id: uuid("id").primaryKey().defaultRandom(),
    agentId: text("agent_id").notNull(),
    settingKey: text("setting_key").notNull(),
    settingValue: text("setting_value"),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow(),
});

export const cache = pgTable("cache", {
    key: text("key").notNull(),
    agentId: text("agentId").notNull(),
    value: jsonb("value").default({}),
    createdAt: timestamp("createdAt").defaultNow(),
    expiresAt: timestamp("expiresAt"),
});

export const consciousnessStreams = pgTable("consciousness_streams", {
    id: uuid("id").primaryKey().defaultRandom(),
    agentId: text("agent_id"),
    topic: text("topic").notNull(),
    title: text("title").notNull(),
    content: jsonb("content").notNull(),
    status: text("status").notNull(),
    timestamp: timestamp("timestamp").defaultNow(),
});

export const goals = pgTable("goals", {
    id: uuid("id").primaryKey(),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
    userId: uuid("user_id").references(() => accounts.id),
    name: text("name"),
    status: text("status"),
    description: text("description"),
    roomId: uuid("room_id").references(() => rooms.id),
    objectives: jsonb("objectives").notNull().default([]),
});

export const logs = pgTable("logs", {
    id: uuid("id").primaryKey(),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
    userId: uuid("user_id")
        .notNull()
        .references(() => accounts.id),
    body: jsonb("body").notNull().default({}),
    type: text("type").notNull(),
    roomId: uuid("room_id"),
});

export const memories = pgTable("memories", {
    id: uuid("id").primaryKey(),
    type: text("type").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
    content: jsonb("content").notNull(),
    embedding: text("embedding"), // Note: vector type not directly supported in drizzle, using text as placeholder
    userId: uuid("user_id").references(() => accounts.id),
    agentId: uuid("agent_id").references(() => accounts.id),
    roomId: uuid("room_id").references(() => rooms.id),
    unique: boolean("unique").notNull().default(true),
});

export const participants = pgTable("participants", {
    id: uuid("id").primaryKey(),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
    userId: uuid("user_id").references(() => accounts.id),
    roomId: uuid("room_id").references(() => rooms.id),
    userState: text("user_state"),
    lastMessageRead: text("last_message_read"),
});

export const relationships = pgTable("relationships", {
    id: uuid("id").primaryKey(),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
    userA: uuid("user_a")
        .notNull()
        .references(() => accounts.id),
    userB: uuid("user_b")
        .notNull()
        .references(() => accounts.id),
    status: text("status"),
    userId: uuid("user_id")
        .notNull()
        .references(() => accounts.id),
});

export const rooms = pgTable("rooms", {
    id: uuid("id").primaryKey(),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
});

export const streamSettings = pgTable("stream_settings", {
    id: uuid("id").primaryKey().defaultRandom(),
    agentId: text("agent_id"),
    enabled: boolean("enabled").default(false),
    interval: integer("interval").default(15),
    lastRun: timestamp("last_run").defaultNow(),
});

export const tweets = pgTable("tweets", {
    id: uuid("id").primaryKey().defaultRandom(),
    content: text("content").notNull(),
    status: text("status").default("pending"),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow(),
    scheduledFor: timestamp("scheduled_for", { withTimezone: true }),
    sentAt: timestamp("sent_at", { withTimezone: true }),
    error: text("error"),
    agentId: text("agent_id"),
    prompt: text("prompt"),
    homeTimeline: jsonb("home_timeline"),
    newTweetContent: text("new_tweet_content"),
    mediaType: text("media_type").default("text"),
    mediaUrl: text("media_url"),
});

export const agentPrompts = pgTable("agent_prompts", {
    id: uuid("id").primaryKey().defaultRandom(),
    prompt: text("prompt").notNull(),
    agentId: text("agent_id").notNull(),
    version: text("version").notNull(),
    enabled: boolean("enabled").notNull().default(true),
    type: text("type").notNull().default("text"),
});

export type Tweet = InferSelectModel<typeof tweets>;
export type AgentPrompt = InferSelectModel<typeof agentPrompts>;
export type StreamSetting = InferSelectModel<typeof streamSettings>;
export type AgentSetting = InferSelectModel<typeof agentSettings>;
export type Cache = InferSelectModel<typeof cache>;
export type ConsciousnessStream = InferSelectModel<typeof consciousnessStreams>;
export type Goal = InferSelectModel<typeof goals>;
export type Log = InferSelectModel<typeof logs>;
export type Memory = InferSelectModel<typeof memories>;
export type Participant = InferSelectModel<typeof participants>;
