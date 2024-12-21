import { elizaLogger, stringToUuid } from "@ai16z/eliza";
import Together from "together-ai";
import { z } from "zod";
import { oai, client } from "./stream-engine";
import { uploadImage } from "../storage/cdn";
import { logQueries } from "../db/queries";
import { Log } from "../db/schema";
const MediaTweetSchema = z.object({
    tweetText: z
        .string()
        .describe(
            "The text of the tweet. Should be 280 characters or less and only 1 sentence."
        ),
    imagePrompt: z
        .string()
        .describe(
            "The prompt that will be used to generate the image you are describing"
        ),
});

interface MediaTweetResponse {
    url: string;
    tweetText: string;
}

export type MediaTweet = z.infer<typeof MediaTweetSchema>;

/***
 * Generate a media tweet for a given runtime
 * @param runtime - The runtime to generate a media tweet for
 * @param prompt - The prompt to use for the media tweet. Include full details including characteristics of the agent, the topic, and the style of the tweet.
 * @returns The media tweet response
 */
export async function generateMediaTweet(
    prompt?: string,
    agentId?: string,
    provider: string = "together"
): Promise<MediaTweetResponse> {
    try {
        elizaLogger.info(`Generating media tweet for runtime:`);

        // Generate tweet content using instructor
        let tweetContent;
        const imgPrompt = `Generate a tweet and image prompt. For the image prompt, you MUST use the exact format below,
            but replace BOTH bracketed sections with your own creative descriptions.
            The first bracketed section needs a unique main scene (examples provided for inspiration),
            and the second bracketed section needs matching background elements that complement your scene.\n\n*
            A specific image prompt for a da Vinci-style painting study:
                \"A Renaissance-style painting in the authentic style of Leonardo da Vinci, blending techniques from Raphael and Titian. The painting depicts [REPLACE WITH YOUR UNIQUE SCENE]
                Here are examples for inspiration:
                    - a serene landscape with rolling hills and distant villages,
                    - an intimate family portrait in a warmly lit interior,
                    - a close-up study of hands engaged in delicate work,
                    - a young woman's face with an enigmatic smile,
                    - a detailed study of flowing water around rocks,
                    - a modern cityscape where nature intertwines with architecture,
                    - a gathering of scholars in animated discussion,
                    - a portrait of a noble figure with intricate costume details,
                    - an astronaut floating amidst the starts and planets,
                    - a child's face full of wonder and curiosity,
                    - an artist's workshop filled with tools and experiments,
                    - a close-up of mechanical clockwork devices,
                    - or a quiet domestic scene with subtle interactions.
                    Create something original while maintaining da Vinci's characteristic attention to detail and human elements


                The scene is crafted with da Vinci's signature sfumato technique, using translucent layers of paint to blend soft edges seamlessly.
                Figures and elements within the composition are precise yet stylized, avoiding photorealistic details while maintaining anatomical and spatial accuracy.
                Visible brushstrokes and layered pigments create a tactile, painted effect, with soft highlights of lead white and warm undertones of raw sienna for lifelike depth.
                The background is rendered with atmospheric perspective, depicting [REPLACE WITH YOUR OWN BACKGROUND ELEMENTS that complement your scene - examples: misty rolling hills, distant trees, a softly lit interior, etc].
                The color palette is a harmonious blend of muted earthy tones such as ochre, burnt umber, olive green, and ultramarine blue, accented with hints of crimson and gold.
                The lighting is warm and diffused, casting natural shadows that are softly rendered using deep glazes to enhance depth and realism. The painting replicates the texture of Renaissance oil paintings on wood panels, with faint craquelure and a slightly aged varnish sheen.
                Subtle imperfections, like uneven layering of paint, evoke the hand-crafted mastery of da Vinci. Brushstrokes are visible everywhere 100%, enhancing the authentic feel of the piece.
                The composition captures an ethereal, timeless quality, with a focus on soft transitions, organic realism, and painterly details that reflect the Renaissance legacy\""`;

        try {
            tweetContent = await client.chat.completions.create({
                messages: [
                    {
                        role: "system",
                        content: prompt,
                    },
                    {
                        role: "user",
                        content: imgPrompt,
                    },
                ],
                model: "gpt-4o-mini",
                response_model: {
                    schema: MediaTweetSchema,
                    name: "MediaTweet",
                },
            });
        } catch (error) {
            elizaLogger.error("Error generating media tweet:", error);
            throw error;
        }

        elizaLogger.info("tweetContent", tweetContent);

        // Generate image using DALL-E
        let response;
        if (provider === "dall-e-3") {
            response = await oai.images.generate({
                model: "dall-e-3",
                prompt: tweetContent.imagePrompt,
                size: "1024x1024",
                quality: "hd",
                style: "vivid",
            });
        } else {
            const together = new Together({
                apiKey: process.env.TOGETHER_API_KEY,
            });

            response = await together.images.create({
                model: "black-forest-labs/FLUX.1-dev",
                prompt: tweetContent.imagePrompt,
                steps: 10,
                n: 4,
            });
        }

        elizaLogger.info("response", response);

        // log the prompt
        try {
            const logBody = {
                prompt: tweetContent.imagePrompt,
                imagePrompt: tweetContent.imagePrompt,
                textPrompt: prompt,
            };
            elizaLogger.info("logBody", logBody);
            const log: Log = {
                id: stringToUuid(`media_tweet_log_${new Date().getTime()}`),
                userId: agentId,
                body: JSON.stringify(logBody),
                type: "media_tweet",
                createdAt: new Date(),
                roomId: `media_tweet_log`,
            };
            await logQueries.saveLog(log);
        } catch (error) {
            elizaLogger.error("Error logging media tweet prompt:", error);
        }

        const remoteImage = await uploadImage(response.data[0].url);

        return {
            url: remoteImage.url,
            tweetText: tweetContent.tweetText,
        };
    } catch (error) {
        elizaLogger.error("Error generating media tweet:", error);
        throw error;
    }
}
