import type { Hit } from "../types";

interface SearchResultsProps {
    results: Hit[];
    onSelect: (docId: string) => void;
}

export default function SearchResults(props: SearchResultsProps) {
    return (
        <div>
            {
                props.results && props.results.length && <h2>Results</h2>
            }
            <ul>
                {props.results.map(hit => (
                    <li>
                        <a
                            href="#"
                            onClick={(e) => {
                                e.preventDefault();
                                props.onSelect(hit._id);
                            }}
                        >
                            {hit._source.title} ({hit._source.author}, {hit._source.year})
                        </a>
                    </li>
                ))}
            </ul>
        </div>
    );
}
