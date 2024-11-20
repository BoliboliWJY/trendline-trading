def time_number(interval):
    try:
        time_unit = interval[-1]
        time_value = int(interval[:-1])
    except (ValueError, IndexError):
        raise ValueError("Invalid interval format, should be like '5m', '1h'")

    if time_unit == 's':
        return time_value
    elif time_unit == 'm':
        return time_value * 60
    elif time_unit == 'h':
        return time_value * 3600
    elif time_unit == 'd':
        return time_value * 86400
    elif time_unit == 'w':
        return time_value * 604800
    else:
        raise ValueError("Unsupported time unit")
