export default function SpatialMap({ objects, lidar }) {
	const lidarCells = lidar?.grid || [];

	return (
		<section className="spatial-map">
			<div className="section-heading">
				<h2>Spatial Map</h2>
				<span className={lidar?.connected ? "lidar-status connected" : "lidar-status"}>
					LiDAR: {lidar?.connected ? "CONNECTED" : "NOT CONNECTED"}
				</span>
			</div>
			<div className="map-grid">
				<span className="map-axis axis-left">LEFT</span>
				<span className="map-axis axis-right">RIGHT</span>
				{[5, 10, 15, 20, 25, 30].map((distance) => (
					<div
						className="map-ring"
						key={distance}
						style={{ bottom: `${(distance / 30) * 82}%` }}
					>
						<span>{distance}m</span>
					</div>
				))}
				<div className="map-you">YOU</div>
				{objects.map((object, index) => (
					<div
						className={`map-object ${object.risk.toLowerCase()}`}
						key={`${object.id}-${index}`}
						style={{
							left: `${50 + object.x * 42}%`,
							bottom: `${Math.min(82, Math.max(8, (1 - object.distance / 30) * 82))}%`
						}}
						title={`${object.class}, ${object.distance}m`}
					>
						{object.id ?? "?"}
					</div>
				))}
				{lidarCells.map((cell, index) => (
					<div
						className={`lidar-cell ${cell.obstacle ? "obstacle" : "traversable"}`}
						key={`lidar-${index}`}
						style={{
							left: `${50 + (cell.x / 30) * 42}%`,
							bottom: `${Math.min(82, Math.max(8, (1 - Math.abs(cell.y) / 30) * 82))}%`
						}}
						title={`LiDAR x ${cell.x}m, y ${cell.y}m, z ${cell.z}m`}
					/>
				))}
			</div>
		</section>
	);
}
