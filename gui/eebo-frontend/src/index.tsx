import { render } from "solid-js/web";
import 'beercss';
// import App from "./OldEventsApp";
import App from "./App";
import "./index.css"

render(
    () => (
        <App />
    ),
    document.getElementById("root")!
);
