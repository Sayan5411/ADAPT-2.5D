export default function CameraView({
	videoRef,
	cameraStarted,
	objects,
	onStart
}) {
	return (
		<div className="camera-box">
			<video
				ref={videoRef}
				className="camera"
				muted
				playsInline
			/>

			{!cameraStarted && (
				<div className="camera-placeholder">
					<h2>Camera Offline</h2>
					<p>Start the camera to begin detection.</p>
					<button onClick={onStart}>START CAMERA</button>
				</div>
			)}

			{objects.map((object, index) => {
				const riskClass = object.risk.toLowerCase();

				return (
					<div
						key={`${object.id}-${index}`}
						className={`box ${riskClass}`}
						style={{
							left: `${(object.x1 / 640) * 100}%`,
							top: `${(object.y1 / 360) * 100}%`,
							width: `${((object.x2 - object.x1) / 640) * 100}%`,
							height: `${((object.y2 - object.y1) / 360) * 100}%`
						}}
					>
						<span className="label">
							#{object.id ?? "-"} {object.class} {object.distance}m
						</span>
					</div>
				);
			})}
		</div>
	);
}
