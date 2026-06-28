import type { Connect } from "vite";
import "dotenv/config";
import Groq from "groq-sdk";

import { text, serverError } from "../lib/response";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

/**
 * Request shape coming from frontend
 */
export interface Cluster2GroqRequest {
    concept: string;
    points: {
        id: string;
        x: number;
        y: number;
    }[];
    rawText?: string;
}

export function createGroqMiddleware(): Connect.NextHandleFunction {
    return async (req, res, next) => {
        if (!req.url) return next();

        if (!req.url.startsWith("/api/groq")) return next();

        try {
            // Parse request body
            const body = await new Promise<string>((resolve, reject) => {
                let data = "";
                req.on("data", (c) => (data += c));
                req.on("end", () => resolve(data));
                req.on("error", reject);
            });

            const parsed = JSON.parse(body) as Cluster2GroqRequest;

            // Validate request
            if (
                !parsed ||
                typeof parsed.concept !== "string" ||
                !Array.isArray(parsed.points)
            ) {
                throw new TypeError(
                    `Expected { concept: string, points: {id,x,y}[] } but got: ${ body }`
                );
            }

            const results = await analyzeCluster(parsed);

            return text(res, 200, JSON.stringify(results));
        } catch (error) {
            return serverError(res, error);
        }
    };
}

/**
 * Safe JSON parsing fallback for LLM output
 */
function safeJsonParse(raw: string) {
    try {
        return JSON.parse(raw);
    } catch {
        const match = raw.match(/\{[\s\S]*\}/);
        if (!match) throw new Error("No JSON found in model output");
        return JSON.parse(match[0]);
    }
}

/**
 * Core Groq semantic cluster labeling
 */
export async function analyzeCluster(input: Cluster2GroqRequest) {
    const { concept, points, rawText } = input;

    // Keep lightweight context for LLM
    const sampleText = (rawText ?? "").substring(0, 1000);

    const clusterSummary = `
Concept: ${ concept }
Point count: ${ points.length }

Sample text:
${ sampleText }
`;

    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content:
                    "You are an expert in historical English semantics, specializing in Early English Books Online (EEBO). You label semantic usage clusters based on contextual evidence.",
            },
            {
                role: "user",
                content: `
TASK:
Given a cluster of occurrences of a word, identify ONE unified semantic sense.

CLUSTER DATA:
${ clusterSummary }

RULES:
- Only ONE sense
- Do NOT output multiple meanings
- Do NOT give dictionary definitions
- Do NOT provide historical commentary
- Focus only on usage in this cluster

OUTPUT FORMAT (STRICT JSON ONLY):
{
  "sense_name": string (max 8 words),
  "description": string (1–2 sentences)
}
`,
            },
        ],
        model: "llama-3.3-70b-versatile",
        temperature: 0.3,
        response_format: { type: "json_object" },
    });

    const raw = completion.choices[0]?.message?.content;

    if (!raw) {
        throw new Error("No model output");
    }

    return safeJsonParse(raw);
}
