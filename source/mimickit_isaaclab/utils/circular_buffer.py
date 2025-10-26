"""Circular buffer for storing history data.

This module provides a circular buffer implementation for efficiently storing
and accessing historical data in a ring buffer fashion.
"""

import torch


class CircularBuffer:
    """A circular buffer for storing historical data."""

    def __init__(
        self,
        batch_size: int,
        buffer_len: int,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
    ):
        """Initialize the circular buffer.
        
        Args:
            batch_size: Number of parallel buffers (e.g., number of environments)
            buffer_len: Length of the buffer (number of time steps to store)
            shape: Shape of each data element (excluding batch dimension)
            dtype: Data type of the buffer
            device: Device to store the buffer on
        """
        self._buffer = torch.zeros([batch_size, buffer_len] + list(shape), dtype=dtype, device=device)
        self._head = 0

    def get_batch_size(self) -> int:
        """Get the batch size of the buffer."""
        return self._buffer.shape[0]

    def get_buffer_len(self) -> int:
        """Get the length of the buffer."""
        return self._buffer.shape[1]

    def push(self, data: torch.Tensor):
        """Push new data into the buffer.
        
        Args:
            data: Data to push, shape should be [batch_size, ...]
        """
        self._buffer[:, self._head, ...] = data
        n = self.get_buffer_len()
        self._head = (self._head + 1) % n

    def fill(self, batch_idx: torch.Tensor, data: torch.Tensor):
        """Fill the buffer for specific batch indices.
        
        Args:
            batch_idx: Indices of batches to fill
            data: Data to fill, shape should be [num_idx, buffer_len, ...]
        """
        buffer_len = self.get_buffer_len()
        self._buffer[batch_idx, self._head:, ...] = data[:, :buffer_len - self._head, ...]
        self._buffer[batch_idx, :self._head, ...] = data[:, buffer_len - self._head:, ...]

    def get(self, idx: int | torch.Tensor) -> torch.Tensor:
        """Get data at specific index in the buffer.
        
        Args:
            idx: Index (or indices) to retrieve
            
        Returns:
            Data at the specified index
        """
        n = self.get_buffer_len()
        buffer_idx = self._head + idx

        if torch.is_tensor(idx):
            batch_size = self.get_batch_size()
            batch_idx = torch.arange(0, batch_size, device=self._buffer.device)
            buffer_idx = torch.remainder(buffer_idx, n)
            data = self._buffer[batch_idx, buffer_idx]
        else:
            buffer_idx = buffer_idx % n
            data = self._buffer[:, buffer_idx, ...]

        return data

    def get_all(self) -> torch.Tensor:
        """Get all data in the buffer in chronological order.
        
        Returns:
            All buffer data, shape [batch_size, buffer_len, ...]
        """
        if self._head == 0:
            data = self._buffer
        else:
            data_beg = self._buffer[:, self._head:, ...]
            n = self.get_buffer_len()
            end_idx = (self._head + n) % n
            data_end = self._buffer[:, 0:end_idx, ...]
            data = torch.cat([data_beg, data_end], dim=1)
        return data

    def reset(self):
        """Reset the buffer head pointer."""
        self._head = 0

