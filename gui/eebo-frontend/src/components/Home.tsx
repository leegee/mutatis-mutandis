import { CORPUS_END_YEAR, CORPUS_START_YEAR, CORPUS_COUNTS } from "../corpus_config";

export default function Home() {
    const format = new Intl.NumberFormat().format;
    return (

        <article class="no-round no-border large extra-padding extra-margin transparent">
            <div class="padding absolute center middle no-round border surface-container">

                <header>
                    <h1>Corpus Viewer</h1>
                    <h5>Selected short texts from {CORPUS_START_YEAR} - {CORPUS_END_YEAR}</h5>
                </header>

                <div class="large-padding large-margin top-margin bottom-margin">
                    <table>
                        <tbody>
                            <tr>
                                <th>Total documents in DB</th>
                                <td class="number">{format(CORPUS_COUNTS.total_docs)}</td>
                            </tr><tr>
                                <th>Total tokens in DB</th>
                                <td class="number">{format(CORPUS_COUNTS.total_tokens)}</td>
                            </tr><tr>
                                <th>Total documents in corpus</th>
                                <td class="number">{format(CORPUS_COUNTS.total_corpus_docs)}</td>
                            </tr><tr>
                                <th>Total tokens in corpus</th>
                                <td class="number">{format(CORPUS_COUNTS.total_corpus_tokens)}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <div class="s4 margin medium-opacity center">
                    <p>
                        Choose an option from the menu on the left.
                    </p>
                </div>

            </div>
        </article >
    );
}
