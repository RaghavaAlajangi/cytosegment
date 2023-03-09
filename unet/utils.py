def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add suffix
    if magnitude == 1:
        return '%.1f%s' % (num, 'K')
    elif magnitude == 2:
        return '%.1f%s' % (num, 'M')
    elif magnitude == 3:
        return '%.1f%s' % (num, 'B')
    else:
        return '%.1f' % num
