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

export async function analyzeCluster(term: string, text: string) {
    const completion = await groq.chat.completions.create({
        messages: [
            {
                role: "system",
                content: "You are an expert in historical English semantics and cover the whole range of Early English Books Online.",
            },
            {
                role: "user",
                content: `Analyze this cluster of text snippets and identify the sense of the use of the word "${ term }":\n\n${ text }\n\nFocus ONLY on the following. Be concise and evidence-based. Do not write general historical essays or broad definitions. Just return the sense defined by context without preamble or postamble. No need to be polite.`,
            },
        ],
        model: "llama-3.3-70b-versatile", // or mixtral, qwen, etc.
        temperature: 0.3,
    });

    return completion.choices[0]?.message?.content;
}

