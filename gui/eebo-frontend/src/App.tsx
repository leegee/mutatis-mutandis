import { createSignal } from "solid-js";
import SearchForm from "./components/SearchForm";
import SearchResults from "./components/SearchResults";
import DocumentView from "./components/DocumentView";

import styles from './App.module.css';

export default function App() {
  const [results, setResults] = createSignal([]);
  const [selectedDoc, setSelectedDoc] = createSignal(null);

  return (
    <main>
      <section class={styles.masthead}>
        <h1>EEBO Search</h1>
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
    </main >
  );
}
