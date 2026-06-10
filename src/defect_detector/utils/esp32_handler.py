"""
Formats responses specifically for ESP32 LCD display
ESP32 has limited processing, so we send pre-formatted text
"""

def format_for_esp32(results, enabled_detections):
    """
    Convert model results into ESP32-friendly format
    
    Returns:
        dict with 'lines' - list of strings for LCD display
                   'led_state' - which LED to light up
    """
    lines = []
    led_state = 'green'  # Default
    
    # Line 1: Header
    lines.append("COFFEE ANALYSIS")
    
    # Foreign Matter Detection
    if 'foreign_matter' in results:
        fm = results['foreign_matter']
        if fm['has_foreign_matter']:
            lines.append(f"FOREIGN: {fm['foreign_count']} found!")
            led_state = 'red'
        else:
            lines.append("Foreign: None")
    
    # Quality Detection
    if 'quality' in results:
        quality = results['quality']['prediction']
        lines.append(f"Quality: {quality}")
        
        # Set LED based on quality
        if quality in ['Grade C', 'Grade D', 'Grade E']:
            led_state = 'yellow'
        elif quality in ['Grade A', 'Grade B']:
            if led_state != 'red':
                led_state = 'green'
    
    # Bean Type Detection
    if 'bean_type' in results:
        bean_type = results['bean_type']['prediction']
        lines.append(f"Type: {bean_type}")
    
    # Footer
    lines.append("================")
    
    return {
        'lines': lines,
        'led_state': led_state,
        'display_text': '\n'.join(lines)  # For LCD that accepts full text
    }