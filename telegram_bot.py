"""
TruthLens AI — Telegram Bot
============================
Send a video to the bot and get deepfake analysis results!
"""

import os
import sys
import tempfile
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ─── Your Bot Token ───
BOT_TOKEN = "8712600195:AAHZVQiTm9UKc4I8_eVbSYw3_vywogSfebE"  # ← Paste your new token here

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Add ffmpeg to path
FFMPEG_PATH = r"C:\Users\gujju\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"
if os.path.exists(FFMPEG_PATH):
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH


# ─── Start Command ───
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔍 *Welcome to TruthLens AI!*\n\n"
        "I can detect deepfake videos using 5 AI signals!\n\n"
        "📹 Just send me a video and I'll analyze it!\n\n"
        "Signals I use:\n"
        "🎵 SyncNet Lip-Sync\n"
        "🎨 Face Texture\n"
        "👁️ Blink Pattern\n"
        "👄 Lip Reader\n"
        "🎙️ Voice Clone\n\n"
        "Send a video to get started! 🚀",
        parse_mode='Markdown'
    )


# ─── Help Command ───
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🔍 *TruthLens AI Help*\n\n"
        "Commands:\n"
        "/start — Welcome message\n"
        "/help — Show this help\n\n"
        "Just send any video and I'll analyze it!\n"
        "Supported formats: MP4, AVI, MOV\n\n"
        "⚠️ Note: Analysis takes 1-2 minutes",
        parse_mode='Markdown'
    )


# ─── Video Handler ───
async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Get video file
    video = update.message.video or update.message.document

    if not video:
        await update.message.reply_text("❌ Please send a video file!")
        return

    # Check file size (max 50MB)
    if video.file_size > 50 * 1024 * 1024:
        await update.message.reply_text(
            "❌ Video too large! Please send a video under 50MB."
        )
        return

    # Send processing message
    processing_msg = await update.message.reply_text(
        "⏳ *Analyzing your video...*\n\n"
        "🔍 Running 5-signal deepfake detection\n"
        "This may take 1-2 minutes...",
        parse_mode='Markdown'
    )

    try:
        # Download video
        file = await context.bot.get_file(video.file_id)
        tmp = tempfile.mktemp(suffix=".mp4")
        await file.download_to_drive(tmp)

        # Update message
        await processing_msg.edit_text(
            "⏳ *Analyzing your video...*\n\n"
            "✅ Video downloaded\n"
            "🔍 Running AI analysis...",
            parse_mode='Markdown'
        )

        # Run TruthLens AI
        from combined_detector import detect
        results = detect(tmp)

        # Cleanup
        os.unlink(tmp)

        if results:
            verdict      = results['verdict']
            final_score  = results['final_score']
            syncnet      = results['syncnet_score']
            texture      = results['texture_score']
            blink        = results['blink_score']
            lip          = results['lip_score']
            voice        = results['voice_score']

            # Verdict emoji
            if verdict == "REAL":
                verdict_emoji = "✅"
                verdict_msg   = "This appears to be a GENUINE video!"
            elif verdict == "DEEPFAKE":
                verdict_emoji = "❌"
                verdict_msg   = "WARNING: This appears to be a DEEPFAKE!"
            else:
                verdict_emoji = "⚠️"
                verdict_msg   = "SUSPICIOUS — Manual review recommended!"

            # Format score bar
            def score_bar(score):
                filled = int(score / 10)
                return "█" * filled + "░" * (10 - filled)

            # Send result
            await processing_msg.edit_text(
                f"{verdict_emoji} *TruthLens AI Results*\n\n"
                f"*VERDICT: {verdict}*\n"
                f"_{verdict_msg}_\n\n"
                f"📊 *Signal Breakdown:*\n"
                f"🎵 SyncNet:     {syncnet:.1f}/100 {score_bar(syncnet)}\n"
                f"🎨 Texture:     {texture:.1f}/100 {score_bar(texture)}\n"
                f"👁️ Blink:       {blink:.1f}/100 {score_bar(blink)}\n"
                f"👄 Lip Reader:  {lip:.1f}/100 {score_bar(lip)}\n"
                f"🎙️ Voice Clone: {voice:.1f}/100 {score_bar(voice)}\n\n"
                f"🎯 *Combined Score: {final_score:.1f}/100*\n\n"
                f"_Powered by TruthLens AI v3.0_",
                parse_mode='Markdown'
            )

        else:
            await processing_msg.edit_text(
                "❌ Analysis failed!\n"
                "Please make sure the video has a visible face."
            )

    except Exception as e:
        await processing_msg.edit_text(
            f"❌ Error during analysis!\n{str(e)[:100]}"
        )
        if os.path.exists(tmp):
            os.unlink(tmp)


# ─── Main ───
def main():
    print("🤖 TruthLens AI Telegram Bot Starting...")
    print(f"🔗 Bot: @TruthLensAI_1bot")

    # Create application
    app = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.VIDEO, handle_video))
    app.add_handler(MessageHandler(filters.Document.VIDEO, handle_video))

    print("✅ Bot is running! Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()