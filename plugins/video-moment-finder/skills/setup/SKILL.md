---
name: setup
description: Help a new or returning VMF user connect, choose or upload their own video, and complete a first useful learning result. Use for plugin setup, getting started, or learning how to use Video Moment Finder.
---

# Get to a useful moment

Aim for one completed result with the user's own source, not a tour of every feature. Discover available schemas. Use the existing account and plugin identity; never request a password or token in chat.

Open `open_workspace` once when available. Its initial result contains the library, actual unit balance and processing tariff. Let the user select a ready video or use Upload video. Do not immediately call the opener again. If the host cannot render it, use `list_videos` and VMF's supported web upload flow. A presigned URL alone is not a completed upload.

- **Ready video:** use the selection already attached to the conversation, or clarify only when ambiguous. Ask what they want to understand, or offer a guide for one important idea. Begin with a bounded excerpt. Follow the study-guide skill and [shared evidence rules](../../references/evidence-format.md), then use the [native view contract](../../references/native-views.md). End with one useful next action: select a confusing moment, try a card or start the tutor.
- **No usable video:** point to Upload video. The user selects a file, confirms their rights and sees the current cost before choosing Upload and process. The workspace transfers bytes and completes the same upload once. Queued/processing is a real wait. Use free status checks; do not create another indexing job to diagnose a failure.
- **Their video is on YouTube:** prefer the original file. Otherwise explain YouTube Studio → Content → video menu → Download, then upload that file. The workspace includes these instructions. A YouTube URL is not a downloadable source, and a host file ID is not a VMF UUID. Do not instruct an agent to run yt-dlp or ask for YouTube cookies.
- **No allowance:** explain the actual units and whether the backend reports a usable trial. Do not promise a disabled trial, grant units, promote an upgrade or link to checkout. Offer sufficient cached evidence or supplied text for a clearly labeled chat result. Reconnection does not reset an allowance.
- **Expired connection:** use the host's reconnect flow and pause calls. The updated VMF tool consent screen requires one reconnect for older connections.

Keep source playback and clickable transcript passages beside the learning view. Transcript/frame retrieval uses the displayed tariffs; library, status and playback refresh are free. The tutor stays in chat and asks one question at a time. Offline HTML, CSV and supported PowerPoint exports remain available. Views belong to this conversation; do not promise permanent learner profiles or cross-chat progress tracking.
