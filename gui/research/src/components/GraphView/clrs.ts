export const typeColors: Record<string, { hue: number }> = {
	concept: { hue: 308 },
	lexeme: { hue: 120 },
	motif: { hue: 283 },
	animal: { hue: 43 },
	person: { hue: 23 },
	evidence: { hue: 204 },
	source: { hue: 244 },
	quote: { hue: 220 },
	group: { hue: 190 },
};

export function hueForType(type: string): number {
	if (typeColors[type]) return typeColors[type].hue;
	// stable fallback hue for any type not explicitly styled
	let hash = 0;
	for (let i = 0; i < type.length; i++) hash = (hash * 31 + type.charCodeAt(i)) % 360;
	return hash;
}
