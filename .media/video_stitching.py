from pathlib import Path
from moviepy import (
    VideoFileClip,
    ImageClip,
    ColorClip,
    clips_array,
    vfx,
)

def stitch_videos_grid(video_paths, output_path="stitched_video.gif", grid_size=None, target_size=(480, 480)):
    """
    Stitches multiple videos into a grid, looping shorter videos to match the longest one.

    Args:
        video_paths (list): A list of paths to the video files.
        output_path (str): The path to save the output video file.
        grid_size (tuple): A tuple (rows, cols) specifying the grid layout.
                           If None, it will try to create a square-like grid.
        target_size (tuple): Target size (width, height) for each video cell.
    """
    # Load all clips (videos or images)
    clips = []
    for p in video_paths:
        path_str = str(p)
        # Check if it's an image file
        if path_str.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            # Load as ImageClip (no duration yet, will set it later)
            clips.append(ImageClip(path_str))
        else:
            # Load as VideoFileClip
            clips.append(VideoFileClip(path_str))

    # Find the duration of the longest video clip
    # For images without duration, we'll set them to match the longest video
    video_durations = [clip.duration for clip in clips if hasattr(clip, 'duration') and clip.duration is not None]
    if video_durations:
        max_duration = max(video_durations)
    else:
        # If all are images, default to 5 seconds
        max_duration = 5.0

    # Set duration for image clips
    for i, clip in enumerate(clips):
        if isinstance(clip, ImageClip):
            clips[i] = clip.with_duration(max_duration)

    # Resize all clips to exactly the same size (square)
    # Strategy: Scale to fit within target size, then add black padding
    target_w, target_h = target_size
    resized_clips = []
    for clip in clips:
        # Calculate scale to FIT within target size (maintaining aspect ratio)
        scale = min(target_w / clip.w, target_h / clip.h)
        new_w = int(clip.w * scale)
        new_h = int(clip.h * scale)

        # Resize the clip
        resized = clip.resized(width=new_w, height=new_h)

        # Add black padding to make it exactly target_size
        pad_left = (target_w - new_w) // 2
        pad_right = target_w - new_w - pad_left
        pad_top = (target_h - new_h) // 2
        pad_bottom = target_h - new_h - pad_top

        if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
            resized = resized.with_effects([
                vfx.Margin(
                    left=pad_left,
                    right=pad_right,
                    top=pad_top,
                    bottom=pad_bottom,
                    color=(0, 0, 0)
                )
            ])

        # Add a thin border (1px on each side) for visual separation
        border_width = 1
        resized = resized.with_effects([
            vfx.Margin(
                left=border_width,
                right=border_width,
                top=border_width,
                bottom=border_width,
                color=(0, 0, 0)
            )
        ])

        resized_clips.append(resized)

    clips = resized_clips

    # Loop shorter clips to match the duration of the longest one
    looped_clips = []
    for clip in clips:
        if clip.duration < max_duration:
            # Calculate how many times to loop
            n_loops = int(max_duration / clip.duration) + 1
            looped = clip.with_effects([vfx.Loop(n=n_loops)])
            # Trim to exact duration using subclipped
            looped = looped.subclipped(0, max_duration)
            looped_clips.append(looped)
        else:
            looped_clips.append(clip)

    # Arrange clips in a grid
    if grid_size is None:
        import math
        num_videos = len(looped_clips)
        cols = math.ceil(math.sqrt(num_videos))
        rows = math.ceil(num_videos / cols)
    else:
        rows, cols = grid_size
    
    # Fill the grid, leaving empty spots if necessary
    grid = []
    clip_iter = iter(looped_clips)
    for _ in range(rows):
        row = []
        for _ in range(cols):
            try:
                row.append(next(clip_iter))
            except StopIteration:
                # If there are not enough videos to fill the grid, add a black clip with border
                border_width = 1
                placeholder = ColorClip(
                    size=target_size,
                    color=(0, 0, 0),
                    duration=max_duration
                ).with_effects([
                    vfx.Margin(
                        left=border_width,
                        right=border_width,
                        top=border_width,
                        bottom=border_width,
                        color=(0, 0, 0)
                    )
                ])
                row.append(placeholder)
        grid.append(row)


    # Create the final clip by arranging the clips in an array
    final_clip = clips_array(grid)

    # Write the result to a file
    if output_path.endswith(".gif"):
        final_clip.write_gif(output_path, fps=15)
    elif output_path.endswith(".webp"):
        # WebP is a modern format with better compression than GIF
        final_clip.write_gif(output_path, fps=15, program='ffmpeg')
    else:
        # MP4 video with H.264 codec (best for LinkedIn)
        final_clip.write_videofile(output_path, fps=24, codec='libx264',
                                   audio=False, preset='medium')

    # Close the clips to release resources
    for clip in clips:
        clip.close()
    for clip in looped_clips:
        clip.close()
    final_clip.close()

if __name__ == '__main__':
    # --- Hardcoded video list ---
    # TODO: Modify this list to include your video files
    video_files = [
        "Week2-QLearning-FrozenLake/demo.mp4",
        # "Week4-PPO-minigrid/demo.mp4",
        "Week3-DQN-CartPole/logs/dqn/CartPole-v1_1/videos/best-model-dqn-CartPole-v1-step-0-to-step-500.mp4",
        "Assignment1-PPO-Dynamic/demo.mp4",
        "Assignment3-EmbodyRL-Mujoco/demo.mp4",
        "Week11-InverseRL-LunarLander/outputs/expert_demo.mp4",
        # "Week11-InverseRL-LunarLander/outputs/airl_demo.mp4",
        # "Week11-InverseRL-LunarLander/outputs/reward_simple_position.png",
    ]

    # Convert to Path objects
    current_dir = Path(__file__).parent
    video_paths = [current_dir / v for v in video_files]

    # Verify files exist
    missing_files = [v for v in video_paths if not v.exists()]
    if missing_files:
        print(f"Error: The following video files were not found:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease edit the `video_files` list in video_stitching.py")
        exit(1)

    num_videos = len(video_paths)
    print(f"Found {num_videos} videos to stitch")

    # Fixed layout: 5 videos per row
    # Automatically calculate number of rows needed
    cols = 5
    rows = (num_videos + cols - 1) // cols  # Ceiling division
    layout = (rows, cols)

    print(f"Using grid layout: {rows} rows × {cols} columns")
    if num_videos < rows * cols:
        print(f"Will fill remaining {rows * cols - num_videos} slot(s) with gray placeholders")

    # Generate multiple formats
    print("Stitching videos...")

    # 1. Generate MP4 (best for LinkedIn)
    mp4_file = "stitched_video.mp4"
    print(f"\n[1/2] Creating MP4 video (best for LinkedIn)...")
    stitch_videos_grid(video_paths, mp4_file, grid_size=layout)
    print(f"✓ Successfully created {mp4_file}")

    # 2. Generate GIF (for compatibility)
    gif_file = "stitched_video.gif"
    print(f"\n[2/2] Creating GIF (for compatibility)...")
    stitch_videos_grid(video_paths, gif_file, grid_size=layout)
    print(f"✓ Successfully created {gif_file}")

    print(f"\nDone! Use {mp4_file} for LinkedIn (better quality & smaller size)")

