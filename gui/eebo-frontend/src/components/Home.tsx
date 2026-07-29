import { CORPUS_END_YEAR, CORPUS_START_YEAR } from "../corpus_config";

export default function Home() {
    return (

        <article class="no-round border large extra-padding extra-margin">
            <div class="padding absolute center middle">

                <header>
                    <h1>Corpus Viewer</h1>
                    <h5>Selected short texts from {CORPUS_START_YEAR} - {CORPUS_END_YEAR}</h5>
                </header>

                <div class="s4 margin">
                    <p>
                        Choose an option from the menu on the left.
                    </p>
                </div>

            </div>
        </article >
    );
}
