import NeighbourhoodBrowser from "~/components/NeighbourhoodBrowser/NeighbourhoodBrowser";

console.log("browser route loaded", NeighbourhoodBrowser);
console.log("component type", typeof NeighbourhoodBrowser);

export default function NeighbourhoodBrowserRoute() {
	console.log("browser page rendered");
	return (
		<>
			<h1>OK</h1>
			<NeighbourhoodBrowser />
		</>
	);
}