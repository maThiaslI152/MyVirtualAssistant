import React, { useState, useRef } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  Slider,
  Card,
  CardContent,
} from '@mui/material';
import {
  Mic,
  Stop,
  PlayArrow,
  VolumeUp,
  Translate,
  Assessment,
} from '@mui/icons-material';

interface VoiceAnalysis {
  text: string;
  language: string;
  confidence: number;
  duration: number;
  metadata: {
    duration: number;
    sample_rate: number;
    channels: number;
    samples: number;
  };
}

const VoiceProcessor: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [text, setText] = useState('');
  const [analysis, setAnalysis] = useState<VoiceAnalysis | null>(null);
  const [generatedAudio, setGeneratedAudio] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [volume, setVolume] = useState(1);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        setAudioUrl(audioUrl);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start recording');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const analyzeAudio = async () => {
    if (!audioUrl) return;

    try {
      setLoading(true);
      setError('');

      const response = await fetch(audioUrl);
      const audioBlob = await response.blob();

      const formData = new FormData();
      formData.append('file', audioBlob, 'audio.wav');

      const analysisResponse = await fetch('/api/voice/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!analysisResponse.ok) {
        throw new Error('Failed to analyze audio');
      }

      const data = await analysisResponse.json();
      setAnalysis(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const transcribeAudio = async () => {
    if (!audioUrl) return;

    try {
      setLoading(true);
      setError('');

      const response = await fetch(audioUrl);
      const audioBlob = await response.blob();

      const formData = new FormData();
      formData.append('file', audioBlob, 'audio.wav');

      const transcribeResponse = await fetch('/api/voice/transcribe', {
        method: 'POST',
        body: formData,
      });

      if (!transcribeResponse.ok) {
        throw new Error('Failed to transcribe audio');
      }

      const data = await transcribeResponse.json();
      setText(data.text);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const generateSpeech = async () => {
    if (!text) return;

    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/voice/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate speech');
      }

      const data = await response.json();
      setGeneratedAudio(data.audio);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
        <Box sx={{ flex: '1 1 45%', minWidth: 300 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Audio Analysis
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
              <Button
                variant="contained"
                color={isRecording ? 'error' : 'primary'}
                startIcon={isRecording ? <Stop /> : <Mic />}
                onClick={isRecording ? stopRecording : startRecording}
              >
                {isRecording ? 'Stop Recording' : 'Start Recording'}
              </Button>
              {audioUrl && (
                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  onClick={() => {
                    const audio = new Audio(audioUrl);
                    audio.volume = volume;
                    audio.play();
                  }}
                >
                  Play Recording
                </Button>
              )}
            </Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <VolumeUp />
              <Slider
                value={volume}
                onChange={(_, value) => setVolume(value as number)}
                min={0}
                max={1}
                step={0.1}
                sx={{ flex: 1 }}
              />
            </Box>
            {audioUrl && (
              <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                <Button
                  variant="contained"
                  startIcon={<Assessment />}
                  onClick={analyzeAudio}
                  disabled={loading}
                >
                  Analyze
                </Button>
                <Button
                  variant="contained"
                  startIcon={<Translate />}
                  onClick={transcribeAudio}
                  disabled={loading}
                >
                  Transcribe
                </Button>
              </Box>
            )}
          </Paper>
        </Box>
        <Box sx={{ flex: '1 1 45%', minWidth: 300 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Text to Speech
            </Typography>
            <TextField
              fullWidth
              multiline
              rows={4}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text to convert to speech..."
              variant="outlined"
              sx={{ mb: 2 }}
            />
            <Button
              variant="contained"
              startIcon={<VolumeUp />}
              onClick={generateSpeech}
              disabled={loading || !text}
            >
              Generate Speech
            </Button>
          </Paper>
        </Box>
        {loading && (
          <Box sx={{ width: '100%', display: 'flex', justifyContent: 'center' }}>
            <CircularProgress />
          </Box>
        )}
        {error && (
          <Box sx={{ width: '100%' }}>
            <Alert severity="error">{error}</Alert>
          </Box>
        )}
        {analysis && (
          <Box sx={{ flex: '1 1 100%', p: 2 }}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Analysis Results
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                <Box sx={{ flex: '1 1 45%', minWidth: 200 }}>
                  <Typography variant="subtitle1">Audio Information</Typography>
                  <List>
                    <ListItem>
                      <ListItemText
                        primary="Duration"
                        secondary={`${analysis.duration.toFixed(2)} seconds`}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Sample Rate"
                        secondary={`${analysis.metadata.sample_rate} Hz`}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText
                        primary="Channels"
                        secondary={analysis.metadata.channels}
                      />
                    </ListItem>
                  </List>
                </Box>
                <Box sx={{ flex: '1 1 45%', minWidth: 200 }}>
                  <Typography variant="subtitle1">Transcription</Typography>
                  <Card>
                    <CardContent>
                      <Typography variant="body1">{analysis.text}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        Language: {analysis.language} (Confidence:{' '}
                        {(analysis.confidence * 100).toFixed(1)}%)
                      </Typography>
                    </CardContent>
                  </Card>
                </Box>
              </Box>
            </Paper>
          </Box>
        )}
        {generatedAudio && (
          <Box sx={{ flex: '1 1 100%', p: 2 }}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Generated Speech
              </Typography>
              <Button
                variant="contained"
                startIcon={<PlayArrow />}
                onClick={() => {
                  const audio = new Audio(generatedAudio);
                  audio.volume = volume;
                  audio.play();
                }}
              >
                Play Generated Speech
              </Button>
            </Paper>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default VoiceProcessor; 