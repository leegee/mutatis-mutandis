import { render } from "solid-js/web";
import { Router, Route } from "@solidjs/router";
import "beercss";

import "./index.css";

import App from "./App";
import { routes } from "./routes";

render(
    () => (
        <Router root={App}>
            {routes.map(r => (
                <Route path={r.path} component={r.component} />
            ))}
        </Router>
    ),
    document.getElementById("root")!
);
