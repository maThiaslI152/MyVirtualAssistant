import React, { useState, useEffect } from 'react';
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
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
} from '@mui/material';
import {
  Translate,
  Language,
  Assessment,
  SentimentSatisfied,
  SentimentDissatisfied,
  SentimentNeutral,
} from '@mui/icons-material';

interface LanguageAnalysis {
  language: string;
  confidence: number;
  tokens: string[];
  entities: Array<{
    text: string;
    label: string;
    start: number;
    end: number;
  }>;
  sentiment: {
    label: string;
    score: number;
  };
}

interface SupportedLanguage {
  code: string;
  name: string;
}

const LanguageProcessor: React.FC = () => {
  const [text, setText] = useState('');
  const [targetLanguage, setTargetLanguage] = useState('');
  const [analysis, setAnalysis] = useState<LanguageAnalysis | null>(null);
  const [translatedText, setTranslatedText] = useState('');
  const [supportedLanguages, setSupportedLanguages] = useState<SupportedLanguage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchSupportedLanguages();
  }, []);

  const fetchSupportedLanguages = async () => {
    try {
      const response = await fetch('/api/language/supported');
      if (!response.ok) {
        throw new Error('Failed to fetch supported languages');
      }
      const data = await response.json();
      setSupportedLanguages(data.languages);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    }
  };

  const analyzeText = async () => {
    if (!text) return;

    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/language/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze text');
      }

      const data = await response.json();
      setAnalysis(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const translateText = async () => {
    if (!text || !targetLanguage) return;

    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/language/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          target_language: targetLanguage,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to translate text');
      }

      const data = await response.json();
      setTranslatedText(data.translated_text);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentIcon = (label: string) => {
    switch (label.toLowerCase()) {
      case 'positive':
        return <SentimentSatisfied color="success" />;
      case 'negative':
        return <SentimentDissatisfied color="error" />;
      default:
        return <SentimentNeutral color="warning" />;
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
        <Box sx={{ flex: '1 1 45%', minWidth: 300 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Text Analysis
            </Typography>
            <TextField
              fullWidth
              multiline
              rows={4}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text to analyze..."
              variant="outlined"
              sx={{ mb: 2 }}
            />
            <Button
              variant="contained"
              startIcon={<Assessment />}
              onClick={analyzeText}
              disabled={loading || !text}
            >
              Analyze Text
            </Button>
          </Paper>
        </Box>
        <Box sx={{ flex: '1 1 45%', minWidth: 300 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Translation
            </Typography>
            <TextField
              fullWidth
              multiline
              rows={4}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text to translate..."
              variant="outlined"
              sx={{ mb: 2 }}
            />
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Target Language</InputLabel>
              <Select
                value={targetLanguage}
                onChange={(e) => setTargetLanguage(e.target.value)}
                label="Target Language"
              >
                {supportedLanguages.map((lang) => (
                  <MenuItem key={lang.code} value={lang.code}>
                    {lang.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Button
              variant="contained"
              startIcon={<Translate />}
              onClick={translateText}
              disabled={loading || !text || !targetLanguage}
            >
              Translate
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
                <Box sx={{ flex: '1 1 30%', minWidth: 200 }}>
                  <Typography variant="subtitle1">
                    <Language sx={{ mr: 1 }} />
                    Language Detection
                  </Typography>
                  <List>
                    <ListItem>
                      <ListItemText
                        primary={analysis.language}
                        secondary={`Confidence: ${(analysis.confidence * 100).toFixed(1)}%`}
                      />
                    </ListItem>
                  </List>
                </Box>
                <Box sx={{ flex: '1 1 30%', minWidth: 200 }}>
                  <Typography variant="subtitle1">
                    <Assessment sx={{ mr: 1 }} />
                    Sentiment Analysis
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    {getSentimentIcon(analysis.sentiment.label)}
                    <Typography>
                      {analysis.sentiment.label} (
                      {(analysis.sentiment.score * 100).toFixed(1)}%)
                    </Typography>
                  </Box>
                </Box>
                <Box sx={{ flex: '1 1 30%', minWidth: 200 }}>
                  <Typography variant="subtitle1">Named Entities</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {analysis.entities.map((entity, index) => (
                      <Chip
                        key={index}
                        label={`${entity.text} (${entity.label})`}
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Box>
              </Box>
            </Paper>
          </Box>
        )}
        {translatedText && (
          <Box sx={{ flex: '1 1 100%', p: 2 }}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Translated Text
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={4}
                value={translatedText}
                InputProps={{
                  readOnly: true,
                }}
                variant="outlined"
              />
            </Paper>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default LanguageProcessor; 