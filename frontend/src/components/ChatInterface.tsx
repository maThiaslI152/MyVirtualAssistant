import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Divider,
} from '@mui/material';
import {
  Send as SendIcon,
  AttachFile as AttachFileIcon,
  AccountCircle as UserIcon,
  SmartToy as BotIcon,
  CallSplit as BranchIcon,
} from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  files?: File[];
  branchId?: string;
}

interface ChatBranch {
  id: string;
  parentMessageId: string;
  messages: Message[];
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      text: 'Hello! I am Owlynn, your AI assistant. How can I help you today?',
      sender: 'bot',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [branches, setBranches] = useState<ChatBranch[]>([]);
  const [currentBranch, setCurrentBranch] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: (acceptedFiles) => {
      const newMessage: Message = {
        id: Date.now().toString(),
        text: `Uploaded ${acceptedFiles.length} file(s)`,
        sender: 'user',
        timestamp: new Date(),
        files: acceptedFiles,
      };
      setMessages((prev) => [...prev, newMessage]);
    },
  });

  const handleSend = () => {
    if (input.trim()) {
      const newMessage: Message = {
        id: Date.now().toString(),
        text: input,
        sender: 'user',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, newMessage]);
      setInput('');

      // Simulate bot response
      setTimeout(() => {
        const botResponse: Message = {
          id: (Date.now() + 1).toString(),
          text: 'I received your message. This is a test response.',
          sender: 'bot',
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, botResponse]);
      }, 1000);
    }
  };

  const handleCreateBranch = (messageId: string) => {
    const parentMessage = messages.find((m) => m.id === messageId);
    if (parentMessage) {
      const newBranch: ChatBranch = {
        id: Date.now().toString(),
        parentMessageId: messageId,
        messages: [parentMessage],
      };
      setBranches((prev) => [...prev, newBranch]);
      setCurrentBranch(newBranch.id);
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box
        sx={{
          flex: 1,
          overflow: 'auto',
          p: 2,
          backgroundColor: 'background.default',
        }}
      >
        <List>
          {messages.map((message) => (
            <React.Fragment key={message.id}>
              <ListItem
                alignItems="flex-start"
                sx={{
                  flexDirection: message.sender === 'user' ? 'row-reverse' : 'row',
                }}
              >
                <ListItemAvatar>
                  <Avatar>
                    {message.sender === 'user' ? <UserIcon /> : <BotIcon />}
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={
                    <Box
                      sx={{
                        display: 'flex',
                        justifyContent:
                          message.sender === 'user' ? 'flex-end' : 'flex-start',
                        alignItems: 'center',
                        gap: 1,
                      }}
                    >
                      <Typography
                        component="span"
                        variant="body1"
                        color="text.primary"
                      >
                        {message.text}
                      </Typography>
                      <IconButton
                        size="small"
                        onClick={() => handleCreateBranch(message.id)}
                      >
                        <BranchIcon />
                      </IconButton>
                    </Box>
                  }
                  secondary={
                    <Typography
                      component="span"
                      variant="body2"
                      color="text.secondary"
                    >
                      {message.timestamp.toLocaleTimeString()}
                    </Typography>
                  }
                />
              </ListItem>
              {message.files && (
                <Box sx={{ pl: 9, pr: 2, mb: 2 }}>
                  {message.files.map((file, index) => (
                    <Typography key={index} variant="body2" color="text.secondary">
                      📎 {file.name}
                    </Typography>
                  ))}
                </Box>
              )}
              <Divider variant="inset" component="li" />
            </React.Fragment>
          ))}
          <div ref={messagesEndRef} />
        </List>
      </Box>

      <Box
        {...getRootProps()}
        sx={{
          p: 2,
          borderTop: 1,
          borderColor: 'divider',
          backgroundColor: 'background.paper',
        }}
      >
        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
            InputProps={{
              endAdornment: (
                <IconButton
                  color="primary"
                  onClick={() => {
                    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
                    fileInput?.click();
                  }}
                >
                  <AttachFileIcon />
                </IconButton>
              ),
            }}
          />
          <IconButton color="primary" onClick={handleSend}>
            <SendIcon />
          </IconButton>
        </Box>
        <input {...getInputProps()} style={{ display: 'none' }} />
        {isDragActive && (
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0, 0, 0, 0.5)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              zIndex: 1000,
            }}
          >
            <Typography variant="h6" color="white">
              Drop files here
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
};

export default ChatInterface; 