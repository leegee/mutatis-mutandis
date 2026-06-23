import type { Connect } from "vite";
import 'dotenv/config';
import Groq from 'groq-sdk';

import { text, serverError } from "../lib/middleware";

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

export function createGroqMiddleware(): Connect.NextHandleFunction {
    return async (req, res, next) => {
        // console.debug("[api/grok] enter");
        if (!req.url) return next();

        const match = req.url.match(/^\/api\/groq/);
        if (!match) return next();

        try {
            const body = await new Promise<string>((resolve, reject) => {
                let data = "";
                req.on("data", (c) => (data += c));
                req.on("end", () => resolve(data));
                req.on("error", reject);
            });

            const parsed = JSON.parse(body);

            if (!parsed || !parsed.term || !parsed.text) {
                throw new TypeError(`Expected {term: string, text: string} not ${ body }`);
            }

            const results = await analyzeCluster(parsed.term, parsed.text);
            // console.log(`[api/grok] returning ${ results }`);

            return text(res, 200, JSON.stringify({ results }));
        }
        catch (error) {
            return serverError(res, error);
        }
    }
}

function safeJsonParse(raw: string) {
    try {
        return JSON.parse(raw);
    } catch {
        // fallback: extract JSON block
        const match = raw.match(/\{[\s\S]*\}/);
        if (!match) throw new Error("No JSON found in model output");
        return JSON.parse(match[0]);
    }
}

export async function analyzeCluster(term: string, textExemplars: string) {
    const text = textExemplars.substring(0, 1_000) // 12_000
    console.log("[groq] term =", term, text);
    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content: "You are an expert in historical English semantics and cover the whole range of Early English Books Online.",
            },
            {
                role: "user",
                content: `You are labeling a semantic sense of a word in a historical corpus.

TASK:
Given multiple snippets where the word "${ term }" appears, identify ONE unified sense.

OUTPUT FORMAT (STRICT JSON ONLY):
{
  "sense_name": string (max 8 words),
  "description": string (1-2 sentences)
}

RULES:
- Do NOT provide multiple senses
- Do NOT give general dictionary definitions
- Do NOT write historical commentary
- Only describe the sense as used in this cluster
- Be specific to the evidence

TEXT:
${ text }`
            },
        ],
        model: "llama-3.3-70b-versatile", // or mixtral, qwen, etc.
        temperature: 0.3,
        response_format: { type: "json_object" },
    });

    const raw = completion.choices[0]?.message?.content;

    if (!raw) throw new Error("No model output");

    return safeJsonParse(raw);
}

