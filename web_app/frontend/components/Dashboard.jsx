export default function Dashboard({ data }) {
	const metrics = [
		["Total Objects", data.counts.total, ""],
		["Danger", data.counts.danger, "danger-card"],
		["Warning", data.counts.warning, "warning-card"],
		["Safe", data.counts.safe, "safe-card"],
		["FPS", data.fps, ""],
		["Resolution", data.inference_size, ""]
	];

	return (
		<section className="dashboard">
			{metrics.map(([label, value, className]) => (
				<div className={`card ${className}`} key={label}>
					<h3>{label}</h3>
					<strong>{value}</strong>
				</div>
			))}
		</section>
	);
}
