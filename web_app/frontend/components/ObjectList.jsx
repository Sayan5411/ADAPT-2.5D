export default function ObjectList({ objects }) {
	return (
		<section className="objects">
			<h2>Detected Objects</h2>
			{objects.length === 0 ? (
				<p>No objects detected.</p>
			) : (
				<div className="object-grid">
					{objects.map((object, index) => (
						<div className="object-card" key={`${object.id}-${index}`}>
							<h3>#{object.id ?? "-"} {object.class}</h3>
							<p>Distance: {object.distance} m</p>
							<p>Confidence: {(object.confidence * 100).toFixed(1)}%</p>
							<p>Risk: <strong>{object.risk}</strong></p>
						</div>
					))}
				</div>
			)}
		</section>
	);
}
