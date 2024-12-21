import { z } from "zod";
import Instructor from "@instructor-ai/instructor";
import OpenAI from "openai";
import { elizaLogger } from "@ai16z/eliza";
import { chunk } from "lodash";
import { cacheQueries } from "../db/queries";

const TrendingSearchSchema = z.object({
    id: z.string(),
    query: z.string(),
    search_volume: z.number(),
    increase_percentage: z.number(),
    trend_breakdown: z.array(z.string()).optional(),
});

const ConnectionSchema = z.object({
    trendingSearchId: z.string(),
    source: z.string(),
    target: z.string(),
    query: z
        .string()
        .describe(
            "The query that can be used to search for more data about the source and target."
        ),
    reason: z
        .string()
        .describe("A clear explanation of why these topics are connected"),
});

const TrendingConnectionsSchema = z.object({
    title: z.string().describe("A short title summarizing all the connections"),
    connections: z
        .array(ConnectionSchema)
        .describe("An array of meaningful connections between trending topics"),
});

export const VitruvianStreamSchema = z.object({
    content: z.object({
        title: z
            .string()
            .describe("A short title summarizing all the connections"),
        nodes: z.array(
            z.object({
                query: z.string(),
                search_volume: z.number(),
                increase_percentage: z.number(),
                trend_breakdown: z.array(z.string()).optional(),
            })
        ),
        connections: z.array(
            z.object({
                source: z.string(),
                target: z.string(),
                reason: z.string(),
            })
        ),
        metadata: z.object({
            timestamp: z.string(),
            total_nodes: z.number(),
            total_connections: z.number(),
        }),
    }),
});

interface TavilySearchResult {
    title: string;
    url: string;
    content: string;
}

const EnrichedConnectionSchema = z.object({
    ...ConnectionSchema.shape,
    sourceData: z.array(
        z.object({
            title: z.string(),
            content: z.string(),
            url: z.string(),
        })
    ),
    targetData: z.array(
        z.object({
            title: z.string(),
            content: z.string(),
            url: z.string(),
        })
    ),
});

const FinalEnrichedConnectionSchema = z.object({
    ...ConnectionSchema.shape,
    improvedConnection: z
        .string()
        .describe(
            "An improved connection based on the original connection and the search data."
        ),
    trendingSearch: TrendingSearchSchema.optional().describe(
        "This will be appended by the system, disregard."
    ),
});

const ImprovedConnectionSchema = z.object({
    connections: z.array(FinalEnrichedConnectionSchema),
});

// Create OpenAI client with instructor
const oai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const client = Instructor({
    client: oai,
    mode: "FUNCTIONS",
});

const TRENDING_PROMPT = `Analyze these trending searches and create meaningful connections between related topics.
Where possible you have been provided with search data for the trending searches to help you make connections.
Focus on identifying:
- Political connections and current events
- Sports and entertainment relationships
- Technology and business links
- Cultural and social media trends
- Breaking news and related stories

For each connection:
1. Choose topics that have a genuine, meaningful relationship
2. Provide a clear, specific reason for the connection
3. Ensure the connection adds value to understanding the trends

Create connections that help tell the story of what's trending and why these topics are related.`;

async function fetchTrendingSearches(): Promise<
    z.infer<typeof TrendingSearchSchema>[]
> {
    try {
        const response = await fetch(
            "https://redux-serpaapi.netlify.app/.netlify/functions/trending-searches"
        );
        const data = await response.json();
        return data.trending_searches.map((search: any, idx) => ({
            query: search.query,
            search_volume: search.search_volume,
            increase_percentage: search.increase_percentage,
            trend_breakdown: search.trend_breakdown,
            id: idx.toString(),
        }));
    } catch (error) {
        elizaLogger.error("Error fetching trending searches:", error);
        throw error;
    }
}

export async function generateSimpleTrendingConnections(): Promise<
    z.infer<typeof TrendingConnectionsSchema>
> {
    try {
        // Fetch trending searches
        const trendingSearches = await fetchTrendingSearches();
        const trendingSearchesWithIds = trendingSearches.map((s, idx) => ({
            ...s,
            id: idx.toString(),
        }));

        // Format trending searches for the prompt
        const trendingContext = trendingSearchesWithIds
            .map((search) => {
                const breakdown = search.trend_breakdown
                    ? `\nRelated terms: ${search.trend_breakdown.join(", ")}`
                    : "";
                return `"ID: ${search.id}  Query: ${search.query}" (Volume: ${search.search_volume}, Increase: ${search.increase_percentage}%)${breakdown}`;
            })
            .join("\n\n");

        // Generate connections using the LLM
        const result = await client.chat.completions.create({
            messages: [
                {
                    role: "system",
                    content:
                        "You are an expert analyst of trending topics and current events. Your task is to identify meaningful connections between trending searches and explain their relationships.",
                },
                {
                    role: "user",
                    content: `${TRENDING_PROMPT}\n\nHere are the trending searches:\n${trendingContext}`,
                },
            ],
            model: "gpt-4o-mini",
            response_model: {
                schema: TrendingConnectionsSchema,
                name: "TrendingConnections",
            },
        });

        elizaLogger.info("Generated trending connections:", result);
        return result;
    } catch (error) {
        elizaLogger.error("Error generating trending connections:", error);
        throw error;
    }
}

async function searchTavily(query: string): Promise<TavilySearchResult[]> {
    try {
        const response = await fetch("https://api.tavily.com/search", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                api_key: process.env.TAVILY_API_KEY,
                query,
                max_results: 3, // Limit results for efficiency
            }),
        });

        const data = await response.json();
        return data.results;
    } catch (error) {
        elizaLogger.error("Error searching Tavily:", error);
        return [];
    }
}

async function enrichConnection(
    connection: z.infer<typeof ConnectionSchema>
): Promise<z.infer<typeof EnrichedConnectionSchema>> {
    const [sourceResults, targetResults] = await Promise.all([
        searchTavily(connection.source + " " + connection.query),
        searchTavily(connection.target + " " + connection.query),
    ]);

    return {
        ...connection,
        sourceData: sourceResults,
        targetData: targetResults,
    };
}

async function enrichConnections(
    connections: Array<z.infer<typeof ConnectionSchema>>
): Promise<Array<z.infer<typeof EnrichedConnectionSchema>>> {
    // Process in chunks to avoid rate limits
    const chunks = chunk(connections, 3);
    const enrichedChunks = await Promise.all(
        chunks.map(async (chunk) => {
            const enrichedConnections = await Promise.all(
                chunk.map((connection) => enrichConnection(connection))
            );
            // Add a small delay between chunks
            await new Promise((resolve) => setTimeout(resolve, 1000));
            return enrichedConnections;
        })
    );

    return enrichedChunks.flat();
}

async function improveConnections(
    enrichedConnections: Array<z.infer<typeof EnrichedConnectionSchema>>
): Promise<Array<z.infer<typeof FinalEnrichedConnectionSchema>>> {
    const IMPROVEMENT_PROMPT = `Analyze these connections between trending topics and their associated search data.
For each connection:
1. Verify if the connection is supported by the search data
2. Enhance the explanation with specific details from the search results
3. Add relevant context that wasn't in the original connection
4. Assign a confidence score based on the supporting evidence

Focus on accuracy and providing concrete details from the search data.`;

    const result = await client.chat.completions.create({
        messages: [
            {
                role: "system",
                content:
                    "You are an expert analyst tasked with improving connections between trending topics using real-time search data.",
            },
            {
                role: "user",
                content: `${IMPROVEMENT_PROMPT}\n\nConnections and search data:\n${JSON.stringify(enrichedConnections, null, 2)}`,
            },
        ],
        model: "gpt-4o-mini",
        response_model: {
            schema: ImprovedConnectionSchema,
            name: "ImprovedConnections",
        },
    });

    return result.connections;
}

export async function generateTrendingConnections(): Promise<
    z.infer<typeof VitruvianStreamSchema>
> {
    const trendingSearches = await fetchTrendingSearches();
    // const cachedConnection = await cacheQueries.get("trending_connections");

    // if (cachedConnection && cachedConnection.length > 0) {
    //     const noneExpiredConnections = cachedConnection.filter(
    //         (c) => c.expiresAt && c.expiresAt > new Date()
    //     );
    //     if (noneExpiredConnections.length > 0) {
    //         return JSON.parse(noneExpiredConnections[0].value as string);
    //     }
    // }

    try {
        // Get initial connections
        const trendingConnections = await generateSimpleTrendingConnections();
        // connections = initialConnections;

        // Enrich and improve connections
        const enrichedConnections = await enrichConnections(
            trendingConnections.connections
        );
        const improvedConnections =
            await improveConnections(enrichedConnections);

        // Transform into VitruvianStream format
        const vitruvianData = {
            content: {
                nodes: trendingSearches.map((search) => ({
                    query: search.query,
                    search_volume: search.search_volume,
                    increase_percentage: search.increase_percentage,
                    trend_breakdown: search.trend_breakdown || [],
                })),
                connections: improvedConnections.map((conn) => ({
                    source: conn.source,
                    target: conn.target,
                    reason: conn.improvedConnection || conn.reason,
                })),
                metadata: {
                    timestamp: new Date().toISOString(),
                    total_nodes: trendingSearches.length,
                    total_connections: improvedConnections.length,
                },
                title: trendingConnections.title,
            },
        };

        return vitruvianData;
    } catch (error) {
        elizaLogger.error(
            "Error generating enhanced trending connections:",
            error
        );
        // Return a minimal valid structure instead of throwing
        return {
            content: {
                nodes: [],
                title: "",
                connections: [],
                metadata: {
                    timestamp: new Date().toISOString(),
                    total_nodes: 0,
                    total_connections: 0,
                },
            },
        };
    }
}

// Example usage:
// const connections = await generateTrendingConnections();
// connections.connections.forEach(c => {
//     console.log(`${c.source} -> ${c.target}: ${c.reason}`);
// });
