import { render } from "solid-js/web";
import 'beercss';
import App from "./App";

render(
    () => (
        <App />
    ),
    document.getElementById("root")!
);
