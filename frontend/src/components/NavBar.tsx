import React from 'react';
import { AppBar, Toolbar, Typography, Box } from '@mui/material';

const NavBar: React.FC = () => {
  return (
    <AppBar position="static" color="default" elevation={0}>
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Owlynn
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default NavBar; 