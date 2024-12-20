import bodyParser from "body-parser";
import cors from "cors";
import express from "express";
import { elizaLogger } from "@ai16z/eliza";
import { messageCompletionFooter } from "@ai16z/eliza";
import { AgentRuntime } from "@ai16z/eliza";
import { Client, IAgentRuntime } from "@ai16z/eliza";
import { settings } from "@ai16z/eliza";
import { createApiRouter } from "./api.ts";
import {
    generateStream,
    streamQueries,
} from "../../redux-extensions/src/index";

export const messageHandlerTemplate =
    // {{goals}}
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

export class DirectClient {
    public app: express.Application;
    private agents: Map<string, AgentRuntime>; // container management
    private server: any; // Store server instance
    public startAgent: Function; // Store startAgent functor

    constructor() {
        elizaLogger.log("DirectClient constructor");
        this.app = express();
        this.app.use(cors());
        this.agents = new Map();

        this.app.use(bodyParser.json());
        this.app.use(bodyParser.urlencoded({ extended: true }));

        const apiRouter = createApiRouter(this.agents, this);
        this.app.use(apiRouter);
    }

    private async startStreamGeneration(runtime: AgentRuntime) {
        const settings = await streamQueries.getStreamSettings(runtime.agentId);
        elizaLogger.info(`Stream settings for ${runtime.agentId}:`, settings);

        if (!runtime) {
            elizaLogger.error("Runtime is required for stream generation");
            return;
        }

        // Check if enough time has passed since last run
        const lastRun = new Date(settings.lastRun);
        const minutesSinceLastRun =
            (Date.now() - lastRun.getTime()) / (60 * 1000);

        if (minutesSinceLastRun < settings.interval) {
            const waitMinutes = settings.interval - minutesSinceLastRun;
            elizaLogger.info(
                `Waiting ${waitMinutes.toFixed(1)} minutes before next stream generation`
            );
            setTimeout(
                () => this.startStreamGeneration(runtime),
                waitMinutes * 60 * 1000
            );
            return;
        }

        const topics = [
            "TERMINAL",
            "PAINTING",
            "BOTANY",
            "SKETCH",
            "ASTRONOMY",
        ];
        const randomTopic = topics[Math.floor(Math.random() * topics.length)];

        await generateStream(runtime, randomTopic);

        // Update last run time in settings
        settings.lastRun = new Date();
        await streamQueries.saveStreamSettings(settings);

        // Schedule next run
        setTimeout(
            () => this.startStreamGeneration(runtime),
            settings.interval * 60 * 1000
        );
    }

    // agent/src/index.ts:startAgent calls this
    public registerAgent(runtime: AgentRuntime) {
        this.agents.set(runtime.agentId, runtime);
        this.startStreamGeneration(runtime);
    }

    public unregisterAgent(runtime: AgentRuntime) {
        this.agents.delete(runtime.agentId);
    }

    public start(port: number) {
        this.server = this.app.listen(port, () => {
            elizaLogger.success(
                `REST API bound to 0.0.0.0:${port}. If running locally, access it at http://localhost:${port}.`
            );
        });

        // Handle graceful shutdown
        const gracefulShutdown = () => {
            elizaLogger.log("Received shutdown signal, closing server...");
            this.server.close(() => {
                elizaLogger.success("Server closed successfully");
                process.exit(0);
            });

            // Force close after 5 seconds if server hasn't closed
            setTimeout(() => {
                elizaLogger.error(
                    "Could not close connections in time, forcefully shutting down"
                );
                process.exit(1);
            }, 5000);
        };

        // Handle different shutdown signals
        process.on("SIGTERM", gracefulShutdown);
        process.on("SIGINT", gracefulShutdown);
    }

    public stop() {
        if (this.server) {
            this.server.close(() => {
                elizaLogger.success("Server stopped");
            });
        }
    }
}

export const DirectClientInterface: Client = {
    start: async (_runtime: IAgentRuntime) => {
        elizaLogger.log("DirectClientInterface start");
        const client = new DirectClient();
        const serverPort = parseInt(settings.SERVER_PORT || "3000");
        client.start(serverPort);
        return client;
    },
    stop: async (_runtime: IAgentRuntime, client?: Client) => {
        if (client instanceof DirectClient) {
            client.stop();
        }
    },
};

export default DirectClientInterface;
