import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import multer from "multer";
import {
    AgentRuntime,
    elizaLogger,
    generateCaption,
    generateImage,
    composeContext,
    generateMessageResponse,
    Content,
    Memory,
    ModelClass,
    stringToUuid,
    messageCompletionFooter,
    validateCharacterConfig,
} from "@ai16z/eliza";
import { REST, Routes } from "discord.js";
import { eq, desc } from "drizzle-orm";
import {
    consciousnessStreams,
    authMiddleware,
    generateStream,
} from "../../redux-extensions/src/index";
import { generateTrendingConnections } from "../../redux-extensions/src/engine/trending-searches";
import { db } from "../../redux-extensions/src/index";
import {
    tweetQueries,
    streamQueries,
    promptQueries,
} from "../../redux-extensions/src/db/queries";

import * as fs from "fs";
import * as path from "path";

const upload = multer({ storage: multer.memoryStorage() });

export function createApiRouter(
    agents: Map<string, AgentRuntime>,
    directClient
) {
    const router = express.Router();

    router.use(cors());
    router.use(bodyParser.json());
    router.use(bodyParser.urlencoded({ extended: true }));
    router.use(authMiddleware as any); // authenticate all requests

    router.get("/", (req, res) => {
        res.send("Welcome, this is the REST API!");
    });

    router.get("/hello", (req, res) => {
        res.json({ message: "Hello World!" });
    });

    router.get("/agents", (req, res) => {
        const agentsList = Array.from(agents.values()).map((agent) => ({
            id: agent.agentId,
            name: agent.character.name,
            clients: Object.keys(agent.clients),
        }));
        res.json({ agents: agentsList });
    });

    router.get("/agents/:agentId", (req, res) => {
        const agentId = req.params.agentId;
        const agent = agents.get(agentId);

        if (!agent) {
            res.status(404).json({ error: "Agent not found" });
            return;
        }

        res.json({
            id: agent.agentId,
            character: agent.character,
        });
    });

    router.post("/agents/:agentId/set", async (req, res) => {
        const agentId = req.params.agentId;
        console.log("agentId", agentId);
        let agent: AgentRuntime = agents.get(agentId);

        // update character
        if (agent) {
            // stop agent
            agent.stop();
            directClient.unregisterAgent(agent);
            // if it has a different name, the agentId will change
        }

        // load character from body
        const character = req.body;
        try {
            validateCharacterConfig(character);
        } catch (e) {
            elizaLogger.error(`Error parsing character: ${e}`);
            res.status(400).json({
                success: false,
                message: e.message,
            });
            return;
        }

        // start it up (and register it)
        agent = await directClient.startAgent(character);
        elizaLogger.log(`${character.name} started`);

        res.json({
            id: character.id,
            character: character,
        });
    });

    router.get("/agents/:agentId/channels", async (req, res) => {
        const agentId = req.params.agentId;
        const runtime = agents.get(agentId);

        if (!runtime) {
            res.status(404).json({ error: "Runtime not found" });
            return;
        }

        const API_TOKEN = runtime.getSetting("DISCORD_API_TOKEN") as string;
        const rest = new REST({ version: "10" }).setToken(API_TOKEN);

        try {
            const guilds = (await rest.get(Routes.userGuilds())) as Array<any>;

            res.json({
                id: runtime.agentId,
                guilds: guilds,
                serverCount: guilds.length,
            });
        } catch (error) {
            console.error("Error fetching guilds:", error);
            res.status(500).json({ error: "Failed to fetch guilds" });
        }
    });

    // Whisper endpoint for audio transcription
    router.post(
        "/:agentId/whisper",
        upload.single("file"),
        async (req: any, res) => {
            const audioFile = req.file;
            const agentId = req.params.agentId;

            if (!audioFile) {
                res.status(400).send("No audio file provided");
                return;
            }

            let runtime = agents.get(agentId);

            if (!runtime) {
                runtime = Array.from(agents.values()).find(
                    (a) =>
                        a.character.name.toLowerCase() === agentId.toLowerCase()
                );
            }

            if (!runtime) {
                res.status(404).send("Agent not found");
                return;
            }

            const formData = new FormData();
            const audioBlob = new Blob([audioFile.buffer], {
                type: audioFile.mimetype,
            });
            formData.append("file", audioBlob, audioFile.originalname);
            formData.append("model", "whisper-1");

            const response = await fetch(
                "https://api.openai.com/v1/audio/transcriptions",
                {
                    method: "POST",
                    headers: {
                        Authorization: `Bearer ${runtime.token}`,
                    },
                    body: formData,
                }
            );

            const data = await response.json();
            res.json(data);
        }
    );

    // Message endpoint
    router.post("/:agentId/message", async (req, res) => {
        const agentId = req.params.agentId;
        const roomId = stringToUuid(
            req.body.roomId ?? "default-room-" + agentId
        );
        const userId = stringToUuid(req.body.userId ?? "user");

        let runtime = agents.get(agentId);

        if (!runtime) {
            runtime = Array.from(agents.values()).find(
                (a) => a.character.name.toLowerCase() === agentId.toLowerCase()
            );
        }

        if (!runtime) {
            res.status(404).send("Agent not found");
            return;
        }

        await runtime.ensureConnection(
            userId,
            roomId,
            req.body.userName,
            req.body.name,
            "direct"
        );

        const text = req.body.text;
        const messageId = stringToUuid(Date.now().toString());

        const content: Content = {
            text,
            attachments: [],
            source: "direct",
            inReplyTo: undefined,
        };

        const userMessage = {
            content,
            userId,
            roomId,
            agentId: runtime.agentId,
        };

        const memory: Memory = {
            id: messageId,
            agentId: runtime.agentId,
            userId,
            roomId,
            content,
            createdAt: Date.now(),
        };

        await runtime.messageManager.createMemory(memory);

        const state = await runtime.composeState(userMessage, {
            agentName: runtime.character.name,
        });

        const context = composeContext({
            state,
            template: messageHandlerTemplate,
        });

        const response = await generateMessageResponse({
            runtime: runtime,
            context,
            modelClass: ModelClass.SMALL,
        });

        const responseMessage = {
            ...userMessage,
            userId: runtime.agentId,
            content: response,
        };

        await runtime.messageManager.createMemory(responseMessage);

        if (!response) {
            res.status(500).send("No response from generateMessageResponse");
            return;
        }

        let message = null as Content | null;

        await runtime.evaluate(memory, state);

        const _result = await runtime.processActions(
            memory,
            [responseMessage],
            state,
            async (newMessages) => {
                message = newMessages;
                return [memory];
            }
        );

        if (message) {
            res.json([message, response]);
        } else {
            res.json([response]);
        }
    });

    // Image generation endpoint
    router.post("/:agentId/image", async (req, res) => {
        const agentId = req.params.agentId;
        const agent = agents.get(agentId);
        if (!agent) {
            res.status(404).send("Agent not found");
            return;
        }

        const images = await generateImage({ ...req.body }, agent);
        const imagesRes: { image: string; caption: string }[] = [];
        if (images.data && images.data.length > 0) {
            for (let i = 0; i < images.data.length; i++) {
                const caption = await generateCaption(
                    { imageUrl: images.data[i] },
                    agent
                );
                imagesRes.push({
                    image: images.data[i],
                    caption: caption.title,
                });
            }
        }
        res.json({ images: imagesRes });
    });

    // Fine-tune endpoints
    router.post("/fine-tune", async (req, res) => {
        try {
            const response = await fetch(
                "https://api.bageldb.ai/api/v1/asset",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-API-KEY": `${process.env.BAGEL_API_KEY}`,
                    },
                    body: JSON.stringify(req.body),
                }
            );

            const data = await response.json();
            res.json(data);
        } catch (error) {
            res.status(500).json({
                error: "Failed to forward request to BagelDB",
                details: error.message,
            });
        }
    });

    router.get("/fine-tune/:assetId", async (req, res) => {
        const assetId = req.params.assetId;
        const downloadDir = path.join(process.cwd(), "downloads", assetId);

        try {
            await fs.promises.mkdir(downloadDir, { recursive: true });
            const fileResponse = await fetch(
                `https://api.bageldb.ai/api/v1/asset/${assetId}/download`,
                {
                    headers: {
                        "X-API-KEY": `${process.env.BAGEL_API_KEY}`,
                    },
                }
            );

            if (!fileResponse.ok) {
                throw new Error(
                    `API responded with status ${fileResponse.status}`
                );
            }

            const fileName =
                fileResponse.headers
                    .get("content-disposition")
                    ?.split("filename=")[1]
                    ?.replace(/"/g, "") || "default_name.txt";

            const arrayBuffer = await fileResponse.arrayBuffer();
            const buffer = Buffer.from(arrayBuffer);
            const filePath = path.join(downloadDir, fileName);
            await fs.promises.writeFile(filePath, buffer);
            const stats = await fs.promises.stat(filePath);

            res.json({
                success: true,
                message: "Single file downloaded successfully",
                downloadPath: downloadDir,
                fileCount: 1,
                fileName: fileName,
                fileSize: stats.size,
            });
        } catch (error) {
            res.status(500).json({
                error: "Failed to download files from BagelDB",
                details: error.message,
                stack: error.stack,
            });
        }
    });

    // Stream endpoints
    router.get("/:agentId/streams", async (req, res) => {
        const agentId = req.params.agentId;
        const limit = parseInt(req.query.limit as string) || 100;

        try {
            const result = await db
                .select()
                .from(consciousnessStreams)
                .where(eq(consciousnessStreams.agentId, agentId))
                .orderBy(desc(consciousnessStreams.timestamp))
                .limit(limit);

            res.json(result);
        } catch (error) {
            elizaLogger.error("Error fetching streams:", error);
            res.status(500).json({ error: "Failed to fetch streams" });
        }
    });

    router.get("/:agentId/streams/settings", async (req, res) => {
        const agentId = req.params.agentId;

        try {
            const settings = await streamQueries.getStreamSettings(agentId);
            res.json(settings);
        } catch (error) {
            elizaLogger.error("Error fetching stream settings:", error);
            res.status(500).json({ error: "Failed to fetch stream settings" });
        }
    });

    router.post("/:agentId/streams/settings", async (req, res) => {
        const agentId = req.params.agentId;
        const { enabled, interval } = req.body;

        try {
            const currentSettings =
                await streamQueries.getStreamSettings(agentId);
            const newSettings = {
                ...currentSettings,
                enabled: enabled ?? currentSettings.enabled,
                interval: interval ?? currentSettings.interval,
            };

            await streamQueries.saveStreamSettings(newSettings);
            res.json(newSettings);
        } catch (error) {
            elizaLogger.error("Error updating stream settings:", error);
            res.status(500).json({ error: "Failed to update stream settings" });
        }
    });

    // Admin endpoints
    router.get("/admin/tweets", async (req, res) => {
        const limit = parseInt(req.query.limit as string) || 200;
        const offset = parseInt(req.query.offset as string) || 0;
        const tweets = await tweetQueries.getTweets(limit, offset);
        res.json(tweets);
    });

    // prompts
    router.get("/prompts", async (req, res) => {
        const prompts = await promptQueries.getPrompts();
        res.json(prompts);
    });

    router.post("/prompts/:id", async (req, res) => {
        const id = req.params.id;
        const settings = req.body;
        const prompt = await promptQueries.updatePrompt(id, settings);
        res.json(prompt);
    });

    router.post("/prompts", async (req, res) => {
        const settings = req.body;
        const prompt = await promptQueries.createPrompt(settings);
        res.json(prompt);
    });

    router.delete("/prompts/:id", async (req, res) => {
        const id = req.params.id;
        await promptQueries.deletePrompt(id);
        res.json({ message: "Prompt deleted" });
    });

    // trending searches endpoint
    router.get("/trending-searches", async (req, res) => {
        const connections = await generateTrendingConnections();
        res.json(connections);
    });

    // Add this to the router in createApiRouter function
    router.post("/:agentId/generateStream", async (req, res) => {
        const agentId = req.params.agentId;
        const { topic } = req.body;

        if (!topic) {
            res.status(400).json({ error: "Topic is required" });
            return;
        }

        const runtime = agents.get(agentId);
        if (!runtime) {
            res.status(404).json({ error: "Agent not found" });
            return;
        }

        try {
            const stream = await generateStream(runtime, topic);
            if (!stream) {
                res.status(500).json({ error: "Failed to generate stream" });
                return;
            }
            res.json(stream);
        } catch (error) {
            elizaLogger.error("Error generating stream:", error);
            res.status(500).json({ error: "Failed to generate stream" });
        }
    });

    return router;
}

export const messageHandlerTemplate =
    `# Action Examples
{{actionExamples}}
(Action examples are for reference only. Do not use the information from them in your response.)

# Knowledge
{{knowledge}}

# Task: Generate dialog and actions for the character {{agentName}}.
About {{agentName}}:
{{bio}}
{{lore}}

{{providers}}

{{attachments}}

# Capabilities
Note that {{agentName}} is capable of reading/seeing/hearing various forms of media, including images, videos, audio, plaintext and PDFs. Recent attachments have been included above under the "Attachments" section.

{{messageDirections}}

{{recentMessages}}

{{actions}}

# Instructions: Write the next message for {{agentName}}.
` + messageCompletionFooter;
