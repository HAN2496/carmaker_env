def calculate_straight_segment_properties(lon, lat):
    """
    Calculate the length and angle for each straight segment between consecutive points.
    """
    segment_lengths = []
    segment_angles = []

    for i in range(len(lon) - 1):
        length = haversine(lon[i], lat[i], lon[i + 1], lat[i + 1])
        angle = calculate_bearing(lon[i], lat[i], lon[i + 1], lat[i + 1])
        segment_lengths.append(length)
        segment_angles.append(angle)

    return segment_lengths, segment_angles

def calculate_curved_segment_properties(lon, lat):
    """
    Approximate curved segments using every three consecutive points and calculate properties.
    """
    curved_segment_properties = []

    for i in range(len(lon) - 2):
        p1 = (lon[i], lat[i])
        p2 = (lon[i + 1], lat[i + 1])
        p3 = (lon[i + 2], lat[i + 2])

        try:
            center, radius = circle_through_three_points(p1, p2, p3)
            angle = np.degrees(np.arccos(1 - haversine(lon[i], lat[i], lon[i + 1], lat[i + 1])**2 / (2 * radius**2)))
            length = np.radians(angle) * radius
            curved_segment_properties.append((radius, angle, length))
        except:
            # Skip segments where calculations fail
            continue

    return curved_segment_properties

# Calculate properties for straight segments
straight_segment_lengths, straight_segment_angles = calculate_straight_segment_properties(lon, lat)

# Calculate properties for curved segments
curved_segment_properties = calculate_curved_segment_properties(lon, lat)

straight_segment_lengths[:5], straight_segment_angles[:5], curved_segment_properties[:5]  # Display first few results
