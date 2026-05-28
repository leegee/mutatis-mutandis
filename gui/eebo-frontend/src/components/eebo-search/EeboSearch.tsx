import { createSignal } from "solid-js";
import SearchForm from "./SearchForm";
import SearchResults from "./SearchResults";
import DocumentView from "./DocumentView";

import styles from './EeboSearch.module.css';

export default function EeboSearch() {
    const [results, setResults] = createSignal([]);
    const [selectedDoc, setSelectedDoc] = createSignal(null);

    return (
        <>
            <section class={styles.masthead}>
                <h2>EEBO Search</h2>
                <SearchForm onSearch={setResults} />
            </section>

            <section class={styles.resultsViewer}>
                <div class={styles.results}>
                    <SearchResults results={results()} onSelect={setSelectedDoc} />
                </div>
                <div class={styles.viewer}>
                    <DocumentView docId={selectedDoc()} />
                </div>
            </section>
        </>
    );
}
