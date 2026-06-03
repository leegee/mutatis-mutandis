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
            }}
        >
            <h3>Settling layout…</h3>
            <progress class="circle light-green-text" />
        </div>

    )
}