import { AgentRuntime, elizaLogger } from "@ai16z/eliza";
import { v4 as uuidv4 } from "uuid";

import Instructor from "@instructor-ai/instructor";
import OpenAI from "openai";
import { z } from "zod";
import { getExampleContent } from "./examples";
import { STREAM_PROMPTS } from "./prompts";
import { streamQueries } from "../db/queries";
import { uploadImage } from "../storage/cdn";
import { ConsciousnessStream } from "../db/schema";
import {
    generateTrendingConnections,
    VitruvianStreamSchema,
} from "./trending-searches";

const PaintingStreamSchema = z.object({
    content: z.object({
        title: z.string(),
        technique: z.string(),
        medium: z.string(),
        imagePrompt: z.string(),
        stage: z.number(),
        palette: z.array(z.string()),
        logs: z.array(z.string()),
        image: z.string().optional(),
    }),
});

const TerminalStreamSchema = z.object({
    content: z.object({
        title: z.string(),
        activeTab: z.enum(["INITIUM"]), //  "ANIMA", "CODEX", "OPUS", "STATUS" disabled
        tabs: z.object({
            INITIUM: z
                .array(z.string())
                .min(15)
                .describe(
                    "Lines of text that describe the current state of the stream in the style of Leonardo da Vinci. Should be at least 15 lines."
                ),
        }),
    }),
});

const AstronomyStreamSchema = z.object({
    content: z.object({
        title: z
            .string()
            .describe(
                "A short memorable title for the stream entry in the style of Leonardo da Vinci."
            ),
        lines: z
            .array(z.string())
            .describe(
                "Lines of text that describe the image and current astronomical data from Nasa. Should be in the style of a terminal session in the style of Leonardo da Vinci."
            ),
        feedStatus: z.string().default("ACTIVE"),
        temporalSync: z.number().default(0.985).optional(),
        deepSpaceCoordinates: z.array(z.number().or(z.string())).optional(),
        apod: z
            .object({
                url: z.string(),
                title: z.string(),
                date: z.string(),
                explanation: z.string(),
            })
            .optional(),
    }),
});

const BotanyStreamSchema = z.object({
    content: z.object({
        experiment: z.object({
            title: z.string(),
            type: z.enum(["spiral", "fractal", "branch", "leaf"]),
            stage: z.number(),
            maxStages: z.number(),
            logs: z.array(z.string()),
            image: z.string().optional(),
            imagePrompt: z.string().optional(),
            visualData: z.object({
                baseSize: z.number(),
                growthFactor: z.number(),
                iterations: z.number(),
                segments: z.array(z.number()),
                branchAngle: z.number(),
                reduction: z.number(),
                color: z.string(),
            }),
        }),
    }),
});

const GalleryStreamSchema = z.object({
    content: z.object({
        title: z.string(),
        year: z.string(),
        imagePrompt: z.string(),
        logs: z.array(z.string()),
        image: z.string().optional(),
    }),
});

const SketchStreamSchema = z.object({
    content: z.object({
        ascii: z.string(),
        title: z.string(),
        animation: z.boolean().default(true),
    }),
});

// Combined schema for all stream types
const schemas = {
    TERMINAL: TerminalStreamSchema,
    ASTRONOMY: AstronomyStreamSchema,
    BOTANY: BotanyStreamSchema,
    GALLERY: GalleryStreamSchema,
    SKETCH: SketchStreamSchema,
    PAINTING: PaintingStreamSchema,
    ANATOMY: z.object({
        content: VitruvianStreamSchema,
    }),
};

async function fetchAPOD() {
    try {
        const start = new Date("1995-06-16").getTime();
        const end = new Date().getTime();
        const randomDate = new Date(start + Math.random() * (end - start));
        const dateStr = randomDate.toISOString().split("T")[0];

        const response = await fetch(
            `https://api.nasa.gov/planetary/apod?api_key=${process.env.NASA_API_KEY}&date=${dateStr}`
        );
        const data = await response.json();

        const url = data.hdurl || data.url;
        return {
            url,
            title: data.title,
            date: data.date,
            explanation: data.explanation,
        };
    } catch (error) {
        elizaLogger.error("Error fetching APOD:", error);
        // Return default fallback data
        return {
            url: "https://apod.nasa.gov/apod/image/2411/LittleplanetGalibier-CamilleNIEL1024.jpg",
            title: "Deep Space View",
            date: new Date().toISOString().split("T")[0],
            explanation: `Winter and summer appear to come on a single night to this stunning little planet.
                         It's planet Earth of course. The digitally mapped, nadir centered panorama covers 360x180 degrees
                         and is composed of frames recorded during January and July from the Col du Galibier in the French Alps.
                         Stars and nebulae of the northern winter (bottom) and summer Milky Way form the complete arcs traversing the rugged,
                         curved horizon. Cars driving along on the road during a summer night illuminate the 2,642 meter high mountain pass,
                         but snow makes access difficult during winter months except by serious ski touring.
                         Cycling fans will recognize the Col du Galibier as one of the most famous climbs in planet Earth's Tour de France`,
        };
    }
}

// Create OpenAI client with instructor
export const oai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

export const client = Instructor({
    client: oai,
    mode: "FUNCTIONS",
});

export async function generateStream(
    runtime: AgentRuntime,
    topic: string
): Promise<ConsciousnessStream> {
    try {
        elizaLogger.info(
            `Generating stream for topic: ${topic} for runtime: ${runtime.agentId}`
        );

        let prompt =
            STREAM_PROMPTS[topic as keyof typeof STREAM_PROMPTS] +
            "Here is an example of what this stream should look like: " +
            getExampleContent(topic);
        let streamEvent;
        let apodData;
        if (topic === "ASTRONOMY") {
            // First generate the base content
            apodData = await fetchAPOD();
            prompt =
                STREAM_PROMPTS[topic as keyof typeof STREAM_PROMPTS] +
                `Nasa image description: ${apodData.explanation}` +
                `Here is an example of what this stream should look like: ${getExampleContent(topic)}`;

            const imageData = apodData.url
                ? [
                      {
                          type: "image_url",
                          image_url: { url: apodData.url },
                      },
                  ]
                : [];
            streamEvent = await client.chat.completions.create({
                messages: [
                    {
                        role: "system",
                        content: `You are Leonardo da Vinci, the Renaissance polymath.
                        Respond in your characteristic style, combining careful observation with artistic and scientific insight.`,
                    },
                    {
                        role: "user",
                        content: [
                            ...imageData,
                            { type: "text", text: STREAM_PROMPTS[topic] },
                        ],
                    },
                ],
                model: "gpt-4o-mini",
                response_model: {
                    schema: schemas[topic],
                    name: "StreamEvent",
                },
            });

            streamEvent.content.apod = apodData;
        } else if (topic === "ANATOMY") {
            // Get trending data using existing function
            const trendingData = await generateTrendingConnections();
            if (trendingData.content.nodes.length === 0) {
                throw new Error("No trending data found");
            }

            streamEvent = {
                content: trendingData.content,
                title: trendingData.content.title,
            };
        } else {
            streamEvent = await client.chat.completions.create({
                messages: [
                    {
                        role: "system",
                        content: `You are Leonardo da Vinci, the Renaissance polymath.
                    Respond in your characteristic style, combining careful observation with artistic and scientific insight.`,
                    },
                    {
                        role: "user",
                        content: prompt,
                    },
                ],
                model: "gpt-4o-mini",
                response_model: {
                    schema: schemas[topic],
                    name: "StreamEvent",
                },
            });
        }

        if (["GALLERY", "PAINTING", "BOTANY"].includes(topic)) {
            const imagePrompt =
                streamEvent.content?.imagePrompt &&
                streamEvent.content.imagePrompt.length > 0
                    ? streamEvent.content.imagePrompt
                    : null;
            elizaLogger.info(`Generating image for topic: ${topic}.`);
            try {
                const response = await oai.images.generate({
                    model: "dall-e-3",
                    prompt: `In the precise style of Leonardo da Vinci's ${
                        topic === "ASTRONOMY"
                            ? "scientific sketches"
                            : topic === "PAINTING"
                              ? "painting techniques"
                              : topic === "BOTANY"
                                ? "botanical studies pencil sketches"
                                : "detailed drawings"
                    }: ${imagePrompt}. Use ${
                        topic === "ASTRONOMY"
                            ? "sepia tones and fine line work"
                            : topic === "PAINTING"
                              ? "sfumato technique and rich colors"
                              : topic === "BOTANY"
                                ? "pencil sketches with botanical details and writings"
                                : "cross-hatching and detailed anatomical precision"
                    }. The image should look exactly like it was DRAWN/PAINTED by Leonardo da Vinci himself.`,
                    size: "1024x1024",
                    quality: "hd",
                    style: "vivid",
                });

                // upload the image to the cdn
                const remoteImage = await uploadImage(response.data[0].url);
                streamEvent.content.image = remoteImage.url;
            } catch (error) {
                elizaLogger.error("Failed to generate image:", error);
            }
        }

        const entry: ConsciousnessStream = {
            id: uuidv4(),
            topic,
            title: `Stream Entry ${new Date().toISOString()}`,
            content: streamEvent,
            status: "ACTIVE",
            timestamp: new Date(),
            agentId: runtime.agentId,
        };

        elizaLogger.info(`Generated stream entry: ${entry.id}`);
        await streamQueries.saveStream(entry);
        return entry;
    } catch (error) {
        elizaLogger.error("Error generating stream:", error);
        // dont throw error, just return empty object
        return;
    }
}
