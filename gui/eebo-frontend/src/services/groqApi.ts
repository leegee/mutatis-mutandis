export async function cluster2groq(term: string, text: string) {
    const body = JSON.stringify({ term, text });
    // console.log(`[groqApi] Send`, body)
    const res = await fetch(`/api/groq`, {
        method: 'POST',
        body,
    });
    if (!res.ok) throw new Error("Failed to load window");
    const json = await res.json();
    const textRv = JSON.stringify(json.results, null, 2);
    console.log(`[groqApi] RV`, textRv)
    return textRv;
}
