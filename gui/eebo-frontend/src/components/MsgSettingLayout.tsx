export default function MsgSettingLayout() {
    return (
        <div
            style={{
                position: "absolute",
                inset: 0,
                display: "flex",
                "flex-direction": "column",
                "align-items": "center",
                "justify-content": "center",
                gap: "1rem",
                "pointer-events": "none",
                "z-index": 999,
            }}
        >
            <h3>Loading data...</h3>
            <progress class="circle" />
        </div>

    )
}