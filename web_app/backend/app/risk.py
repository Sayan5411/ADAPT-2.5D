def calculate_risk(distance, ttc=None):

    if ttc is not None:

        if ttc <= 2:
            return "DANGER"

        if ttc <= 5:
            return "WARNING"

    if distance <= 4:
        return "DANGER"

    if distance <= 10:
        return "WARNING"

    return "SAFE"