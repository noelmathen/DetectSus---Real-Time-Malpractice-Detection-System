import winsound  # For sound alerts on Windows

def play_alarm():
    """Play an alarm sound."""
    try:
        # Windows beep sound
        winsound.Beep(1000, 500)  # Frequency: 1000Hz, Duration: 500ms
        print("Alarm played successfully!")
    except Exception as e:
        print(f"Failed to play alarm: {str(e)}")

def main():
    """Test the alarm functionality."""
    print("Testing the alarm function...")
    play_alarm()

if __name__ == "__main__":
    main()
