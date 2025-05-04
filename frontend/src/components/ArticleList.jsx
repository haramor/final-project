import { Paper, Text, Title, Stack, Anchor } from '@mantine/core';

export function ArticleList({ articles }) {
  if (!articles.length) {
    return <Text>No articles found.</Text>;
  }

  return (
    <Stack spacing="md">
      {articles.map((article, index) => (
        <Paper key={index} p="md" withBorder>
          <Title order={4}>
            <Anchor href={article.url} target="_blank">
              {article.title}
            </Anchor>
          </Title>
          {article.authors && (
            <Text size="sm" color="dimmed">
              {article.authors.join(', ')}
            </Text>
          )}
        </Paper>
      ))}
    </Stack>
  );
} 