import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

async function analyzeCluster(clusterText: string, keyword: string = "liberties") {
  const prompt = `
You are a historian and textual analyst specializing in early modern English documents.

Analyze the following cluster of text snippets around the keyword "${ keyword }":

${ clusterText }

Provide:
1. Main theme of this cluster.
2. Historical and linguistic context (what does "${ keyword }" mean here?).
3. Likely source type (e.g., Bills of Mortality, political pamphlet).
4. Key insights or notable patterns.
5. Short summary (2-3 sentences).

Be precise and evidence-based.
`;

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash', // or gemini-1.5-flash
    contents: [{ parts: [{ text: prompt }] }],
  });

  return response.text;
}

// Example usage
const clusterA = `parishes without the wals...`; // paste your text here

analyzeCluster(clusterA).then(console.log);