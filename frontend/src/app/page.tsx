'use client';

import React, { useState } from 'react';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  useTheme,
  useMediaQuery,
  ListItemButton,
} from '@mui/material';
import {
  Chat as ChatIcon,
  Image as ImageIcon,
  Assignment as TaskIcon,
  Menu as MenuIcon,
} from '@mui/icons-material';
import ChatInterface from '@/components/ChatInterface';
import ImageLibrary from '@/components/ImageLibrary';
import TaskBoard from '@/components/TaskBoard';

const drawerWidth = 240;

export default function Home() {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const [mobileOpen, setMobileOpen] = useState(false);
  const [activeView, setActiveView] = useState<'chat' | 'library' | 'tasks'>('chat');

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const drawer = (
    <Box sx={{ overflow: 'auto' }}>
      <List>
        <ListItem>
          <ListItemButton onClick={() => setActiveView('chat')} selected={activeView === 'chat'}>
            <ListItemIcon>
              <ChatIcon />
            </ListItemIcon>
            <ListItemText primary="Chat" />
          </ListItemButton>
        </ListItem>
        <ListItem>
          <ListItemButton onClick={() => setActiveView('library')} selected={activeView === 'library'}>
            <ListItemIcon>
              <ImageIcon />
            </ListItemIcon>
            <ListItemText primary="Image Library" />
          </ListItemButton>
        </ListItem>
        <ListItem>
          <ListItemButton onClick={() => setActiveView('tasks')} selected={activeView === 'tasks'}>
            <ListItemIcon>
              <TaskIcon />
            </ListItemIcon>
            <ListItemText primary="Tasks" />
          </ListItemButton>
        </ListItem>
      </List>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        {isMobile ? (
          <Drawer
            variant="temporary"
            open={mobileOpen}
            onClose={handleDrawerToggle}
            ModalProps={{
              keepMounted: true,
            }}
            sx={{
              '& .MuiDrawer-paper': {
                boxSizing: 'border-box',
                width: drawerWidth,
              },
            }}
          >
            {drawer}
          </Drawer>
        ) : (
          <Drawer
            variant="permanent"
            sx={{
              '& .MuiDrawer-paper': {
                boxSizing: 'border-box',
                width: drawerWidth,
              },
            }}
            open
          >
            {drawer}
          </Drawer>
        )}
      </Box>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
          backgroundColor: 'background.default',
        }}
      >
        {isMobile && (
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
        )}
        {activeView === 'chat' && <ChatInterface />}
        {activeView === 'library' && <ImageLibrary />}
        {activeView === 'tasks' && <TaskBoard />}
      </Box>
    </Box>
  );
} 