import React, { useState } from 'react';
import {
  Box,
  Card,
  CardMedia,
  CardContent,
  Typography,
  IconButton,
  TextField,
  InputAdornment,
} from '@mui/material';
import {
  Search as SearchIcon,
  Delete as DeleteIcon,
  Info as InfoIcon,
} from '@mui/icons-material';

interface ImageItem {
  id: string;
  url: string;
  name: string;
  timestamp: Date;
  metadata?: {
    size: number;
    type: string;
    dimensions?: {
      width: number;
      height: number;
    };
  };
}

const ImageLibrary: React.FC = () => {
  const [images, setImages] = useState<ImageItem[]>([]);
  const [searchQuery, setSearchQuery] = useState('');

  const handleSearch = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
  };

  const filteredImages = images.filter((image) =>
    image.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleDelete = (id: string) => {
    setImages(images.filter((image) => image.id !== id));
  };

  const handleImageInfo = (image: ImageItem) => {
    // TODO: Implement image info modal
    console.log('Image info:', image);
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ mb: 3 }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Search images..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
        />
      </Box>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 3 }}>
        {filteredImages.map((image) => (
          <Box key={image.id} sx={{ flex: '1 1 300px', maxWidth: { xs: '100%', sm: 'calc(50% - 12px)', md: 'calc(33.33% - 16px)', lg: 'calc(25% - 18px)' } }}>
            <Card>
              <CardMedia
                component="img"
                height="200"
                image={image.url}
                alt={image.name}
              />
              <CardContent>
                <Typography variant="h6" noWrap>
                  {image.name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {new Date(image.timestamp).toLocaleString()}
                </Typography>
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1 }}>
                  <IconButton size="small" onClick={() => handleDelete(image.id)}>
                    <DeleteIcon />
                  </IconButton>
                  <IconButton size="small" onClick={() => handleImageInfo(image)}>
                    <InfoIcon />
                  </IconButton>
                </Box>
              </CardContent>
            </Card>
          </Box>
        ))}
      </Box>
    </Box>
  );
};

export default ImageLibrary; 