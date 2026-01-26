import { createSignal } from "solid-js";
import SearchForm from "./components/SearchForm";
import SearchResults from "./components/SearchResults";
import DocumentView from "./components/DocumentView";

export default function App() {
  const [results, setResults] = createSignal([]);
  const [selectedDoc, setSelectedDoc] = createSignal(null);

  return (
    <main>
      <h1>EEBO Search</h1>

      <SearchForm onSearch={setResults} />

      <SearchResults results={results()} onSelect={setSelectedDoc} />

      {selectedDoc() && <DocumentView docId={selectedDoc()} />}
    </main>
  );
}
