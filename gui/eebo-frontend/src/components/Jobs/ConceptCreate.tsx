// src/components/Jobs/ConceptCreate.tsx

import { createSignal } from "solid-js";
import { API } from "../../services/jobsApi";
import { postJson } from "../../lib/json";

export function ConceptCreate() {
  const [name, setName] = createSignal("");
  const [forms, setForms] = createSignal("");
  const [fps, setFps] = createSignal("");

  const submit = async () => {
    await postJson(API.concepts.create_and_run, {
      concept: name(),
      forms: forms().split(",").map(s => s.trim()),
      false_positives: fps().split(",").map(s => s.trim()),
    }, "Create concept");
  };

  return (
    <section>
      <div class="fieldset fill">
        <div class="row padding">
          <div class="s3 field border large">
            <input type="text" aria-description="Concept name" onInput={(e) => setName(e.currentTarget.value)} />
            <output>Concept name</output>
            <span class="tooltip bottom small">
              <p>
                Collective name for, and first member of, <br />
                the set of lexemes that are used to sample the corpus.
              </p>
              <p>
                Existing names can be replaced here.
              </p>
            </span>
          </div>

          <div class="s3 field border large">
            <textarea aria-description="Concept expressions/forms (CSV)" onInput={(e) => setForms(e.currentTarget.value)} />
            <output>Forms/expressions (comma separated)</output>
            <span class="tooltip bottom small">
              <p>
                A comma-separated list of members of the concept set defined to the left, <br />
                the set of lexemes that are used to sample the corpus.
              </p>
            </span>
          </div>

          <div class="s3 field border large">
            <textarea aria-description="False positives (CSV)" onInput={(e) => setFps(e.currentTarget.value)} />
            <output>False positives (CSV)</output>
            <span class="tooltip bottom medium">
              <p>
                A comma-separated list of lexemes to positively not accept in related vectores<br />
                when sampling the corpus.<br />
                <br />
                <span class="italic">Eg:</span>
                For the form <kbd>king</kbd>, a false positive might be <kbd>sing</kbd>
              </p>
            </span>
          </div>
        </div>
      </div>

      <div class="footer padding margin center-align">
        <button class="primary round extra" onClick={submit}>
          <span>Create or update this Concept</span>
          <i>add_2</i>
        </button>
      </div>
    </section>
  );
}
