#### .. method:: Simulation.SetExcitationPlanewave(IncidenceAngles, sAmplitude=0, pAmplitude=0, Order=0])

	Sets the excitation to be a planewave incident upon the front (first layer specified) of the structure.
	If both tilt angles are specified to be zero, then the planewave is normally incident with the electric field polarized along the x-axis for the p-polarization.
	The phase of each polarization is defined at the origin (z = 0).

#####   Usage::

	S.SetExcitationPlanewave(
		IncidenceAngles=(
			10, # polar angle in [0,180)
			30  # azimuthal angle in [0,360)
		),
		sAmplitude = 0.707+0.707j,
		pAmplitude = 0.707-0.707j,
		Order = 0
	)

#####   Arguments:

	IncidenceAngles
		(pair of numbers) Of the form (phi,theta) with angles in degrees.
		``phi`` and ``theta`` give the spherical coordinate angles of the planewave k-vector.
		For zero angles, the k-vector is assumed to be (0, 0, kz), while the electric field is assumed to be (E0, 0, 0), and the magnetic field is in (0, H0, 0). The angle ``phi`` specifies first the angle by which the E,H,k frame should be rotated (CW) about the y-axis, and the angle ``theta`` specifies next the angle by which the E,H,k frame should be rotated (CCW) about the z-axis.
		Note the different directions of rotations for each angle.
	sAmplitude
		(complex number) The electric field amplitude of the s-polarizations of the planewave.
	pAmplitude
		(complex number) The electric field amplitude of the p-polarizations of the planewave.
	Order
		(integer) An optional positive integer specifying which order (mode index) to excite. Defaults to 0. Refer to :func:`GetBasisSet` for details.

#####   Return values:

	None
	
#### Analysis:
1. Light propagates in the +z direction.
2. Rotating this CW around the y-axis as viewed from +y (of a 70-degree angle) gives a plane of incidence in the xz plane.
3. S-polarized light oscillates (in the electrical domain) in the y-direction, perpendicular to the plane of incidence.
4. S-polarized light oscillates perpendicular to the angle of incidence.
