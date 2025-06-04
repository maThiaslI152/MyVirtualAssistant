import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Paper,
  Grid,
  CircularProgress,
  Alert,
  List,
  ListItem,
  ListItemText,
  Chip,
} from '@mui/material';
import { PlayArrow, Code, Assessment, Build } from '@mui/icons-material';

interface CodeAnalysis {
  complexity: number;
  maintainability: number;
  structure: any;
  dependencies: string[];
  metrics: any;
  suggestions: string[];
}

const CodeProcessor: React.FC = () => {
  const [code, setCode] = useState('');
  const [prompt, setPrompt] = useState('');
  const [analysis, setAnalysis] = useState<CodeAnalysis | null>(null);
  const [generatedCode, setGeneratedCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const analyzeCode = async () => {
    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/code/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze code');
      }

      const data = await response.json();
      setAnalysis(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const generateCode = async () => {
    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/code/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate code');
      }

      const data = await response.json();
      setGeneratedCode(data.code);
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
              Code Analysis
            </Typography>
            <TextField
              fullWidth
              multiline
              rows={10}
              value={code}
              onChange={(e) => setCode(e.target.value)}
              placeholder="Enter code to analyze..."
              variant="outlined"
              sx={{ mb: 2 }}
            />
            <Button
              variant="contained"
              startIcon={<Assessment />}
              onClick={analyzeCode}
              disabled={loading || !code}
            >
              Analyze Code
            </Button>
          </Paper>
        </Box>
        <Box sx={{ flex: '1 1 45%', minWidth: 300 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Code Generation
            </Typography>
            <TextField
              fullWidth
              multiline
              rows={4}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter a description of the code you want to generate..."
              variant="outlined"
              sx={{ mb: 2 }}
            />
            <Button
              variant="contained"
              startIcon={<Code />}
              onClick={generateCode}
              disabled={loading || !prompt}
            >
              Generate Code
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
          <Box sx={{ width: '100%' }}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Analysis Results
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
                <Box sx={{ flex: '1 1 45%', minWidth: 300 }}>
                  <Typography variant="subtitle1">Complexity Metrics</Typography>
                  <List>
                    <ListItem>
                      <ListItemText primary={`Cyclomatic Complexity: ${analysis.complexity}`} />
                    </ListItem>
                    <ListItem>
                      <ListItemText primary={`Maintainability Index: ${analysis.maintainability}`} />
                    </ListItem>
                  </List>
                </Box>
                <Box sx={{ flex: '1 1 45%', minWidth: 300 }}>
                  <Typography variant="subtitle1">Dependencies</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {analysis.dependencies.map((dep, index) => (
                      <Chip key={index} label={dep} />
                    ))}
                  </Box>
                </Box>
                <Box sx={{ width: '100%' }}>
                  <Typography variant="subtitle1">Suggestions</Typography>
                  <List>
                    {analysis.suggestions.map((suggestion, index) => (
                      <ListItem key={index}>
                        <ListItemText primary={suggestion} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              </Box>
            </Paper>
          </Box>
        )}
        {generatedCode && (
          <Box sx={{ width: '100%' }}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Generated Code
              </Typography>
              <TextField
                fullWidth
                multiline
                rows={10}
                value={generatedCode}
                variant="outlined"
                InputProps={{
                  readOnly: true,
                }}
              />
            </Paper>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default CodeProcessor; 