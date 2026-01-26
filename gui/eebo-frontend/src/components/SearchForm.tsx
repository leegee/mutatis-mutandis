import { createSignal } from "solid-js";
import type { Hit } from "../types";

import styles from './SearchForm.module.css';

interface SearchFormProps {
    onSearch: (hits: Hit[]) => void;
}

export default function SearchForm(props: SearchFormProps) {
    const [query, setQuery] = createSignal("");
    const [author, setAuthor] = createSignal("");
    const [year, setYear] = createSignal("");
    const [place, setPlace] = createSignal("");
    const [title, setTitle] = createSignal("");

    const doSearch = async (e: Event) => {
        e.preventDefault();
        const params = new URLSearchParams({
            q: query(),
            author: author(),
            year: year(),
            place: place(),
            title: title()
        });

        console.log('Do search ', JSON.stringify({
            q: query(),
            author: author(),
            year: year(),
            place: place(),
            title: title()
        }));

        const res = await fetch(`http://127.0.0.1:5000/search?${params}`);
        const data = await res.json();
        props.onSearch(data.hits || []);
    };

    return (
        <form class={styles.form} onSubmit={doSearch}>
            <input placeholder="Text" value={query()} onInput={e => setQuery(e.target.value)} />
            <input placeholder="Author" value={author()} onInput={e => setAuthor(e.target.value)} />
            <input placeholder="Year" value={year()} onInput={e => setYear(e.target.value)} />
            <input placeholder="Place" value={place()} onInput={e => setPlace(e.target.value)} />
            <input placeholder="Title" value={title()} onInput={e => setTitle(e.target.value)} />
            <button onClick={doSearch}>Search</button>
        </form>
    );
}
