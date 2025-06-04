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
  Chip,
  Card,
  CardMedia,
  CardContent,
} from '@mui/material';
import {
  Image,
  PhotoCamera,
  Edit,
  Delete,
  Palette,
  Face,
  TextFields,
} from '@mui/icons-material';

interface ImageAnalysis {
  objects: Array<{
    label: string;
    score: number;
    box: number[];
  }>;
  faces: Array<{
    location: number[];
    encoding: number[];
    analysis: any;
  }>;
  text: Array<{
    text: string;
    score: number;
  }>;
  colors: Array<{
    rgb: number[];
    percentage: number;
  }>;
  metadata: {
    format: string;
    mode: string;
    size: number[];
    width: number;
    height: number;
  };
}

const ImageProcessor: React.FC = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [analysis, setAnalysis] = useState<ImageAnalysis | null>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setSelectedImage(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;

    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/image/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: selectedImage }),
      });

      if (!response.ok) {
        throw new Error('Failed to analyze image');
      }

      const data = await response.json();
      setAnalysis(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const generateImage = async () => {
    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/image/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate image');
      }

      const data = await response.json();
      setGeneratedImage(data.image);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const editImage = async () => {
    if (!selectedImage || !prompt) return;

    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/image/edit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: selectedImage,
          prompt,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to edit image');
      }

      const data = await response.json();
      setGeneratedImage(data.image);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const removeBackground = async () => {
    if (!selectedImage) return;

    try {
      setLoading(true);
      setError('');

      const response = await fetch('/api/image/remove-background', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: selectedImage }),
      });

      if (!response.ok) {
        throw new Error('Failed to remove background');
      }

      const data = await response.json();
      setGeneratedImage(data.image);
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
              Image Analysis
            </Typography>
            <input
              type="file"
              accept="image/*"
              onChange={handleImageSelect}
              style={{ display: 'none' }}
              ref={fileInputRef}
            />
            <Button
              variant="contained"
              startIcon={<PhotoCamera />}
              onClick={() => fileInputRef.current?.click()}
              sx={{ mb: 2 }}
            >
              Select Image
            </Button>
            {selectedImage && (
              <Box sx={{ mt: 2 }}>
                <Card>
                  <CardMedia
                    component="img"
                    image={selectedImage}
                    alt="Selected"
                    sx={{ maxHeight: 300, objectFit: 'contain' }}
                  />
                </Card>
                <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                  <Button
                    variant="contained"
                    startIcon={<Image />}
                    onClick={analyzeImage}
                    disabled={loading}
                  >
                    Analyze
                  </Button>
                  <Button
                    variant="contained"
                    startIcon={<Delete />}
                    onClick={removeBackground}
                    disabled={loading}
                  >
                    Remove Background
                  </Button>
                </Box>
              </Box>
            )}
          </Paper>
        </Box>
        <Box sx={{ flex: '1 1 45%', minWidth: 300 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Image Metadata
            </Typography>
            {/* ... image metadata content ... */}
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
                    <Face sx={{ mr: 1 }} />
                    Objects Detected
                  </Typography>
                  <List>
                    {analysis.objects.map((obj, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={obj.label}
                          secondary={`Confidence: ${(obj.score * 100).toFixed(1)}%`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
                <Box sx={{ flex: '1 1 30%', minWidth: 200 }}>
                  <Typography variant="subtitle1">
                    <TextFields sx={{ mr: 1 }} />
                    Text Detected
                  </Typography>
                  <List>
                    {analysis.text.map((text, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={text.text}
                          secondary={`Confidence: ${(text.score * 100).toFixed(1)}%`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
                <Box sx={{ flex: '1 1 30%', minWidth: 200 }}>
                  <Typography variant="subtitle1">
                    <Palette sx={{ mr: 1 }} />
                    Dominant Colors
                  </Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {analysis.colors.map((color, index) => (
                      <Chip
                        key={index}
                        label={`${color.percentage.toFixed(1)}%`}
                        sx={{
                          backgroundColor: `rgb(${color.rgb.join(',')})`,
                          color: 'white',
                        }}
                      />
                    ))}
                  </Box>
                </Box>
              </Box>
            </Paper>
          </Box>
        )}
        {generatedImage && (
          <Box sx={{ flex: '1 1 100%', p: 2 }}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Generated/Edited Image
              </Typography>
              <Card>
                <CardMedia
                  component="img"
                  image={generatedImage}
                  alt="Generated"
                  sx={{ maxHeight: 500, objectFit: 'contain' }}
                />
              </Card>
            </Paper>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default ImageProcessor; 