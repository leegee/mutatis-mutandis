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
                                console.log('Selected', hit._id)
                                props.onSelect(hit._id);
                            }}
                        >
                            [{hit._id}] {hit._source.author || 'anon'} {hit._source.year}: {hit._source.title}
                        </a>
                    </li>
                ))}
            </ul>
        </div>
    );
}
